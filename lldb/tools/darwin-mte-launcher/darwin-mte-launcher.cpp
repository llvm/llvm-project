//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/WithColor.h"
#include <dlfcn.h>
#include <spawn.h>
#include <string.h>
#include <vector>

using namespace llvm;

static std::vector<const char *> get_extended_env(const char *envp[]) {
  // Copy over the current environment.
  std::vector<const char *> new_envp;
  for (const char **e = envp; *e; ++e)
    new_envp.push_back(*e);

  // Python's allocator (pymalloc) is not aware of Memory Tagging Extension
  // (MTE) and crashes.
  // https://bugs.python.org/issue43593
  new_envp.push_back("PYTHONMALLOC=malloc");

  // Collect allocation traces for tagged memory.
  new_envp.push_back("SanitizersAllocationTraces=tagged");

  new_envp.push_back(nullptr);
  return new_envp;
}

int main(int argc, const char *argv[], const char *envp[]) {
  const char *program = argv[1];
  const char **new_args = &argv[1];

  posix_spawnattr_t attr;
  int ret = posix_spawnattr_init(&attr);
  if (ret != 0) {
    WithColor::error() << "posix_spawnattr_init failed\n";
    return EXIT_FAILURE;
  }

  typedef int (*posix_spawnattr_set_use_sec_transition_shims_np_t)(
      posix_spawnattr_t *attr, uint32_t flags);
  posix_spawnattr_set_use_sec_transition_shims_np_t
      posix_spawnattr_enable_memory_tagging_fn =
          (posix_spawnattr_set_use_sec_transition_shims_np_t)dlsym(
              RTLD_DEFAULT, "posix_spawnattr_set_use_sec_transition_shims_np");

  if (!posix_spawnattr_enable_memory_tagging_fn) {
    WithColor::error()
        << "posix_spawnattr_set_use_sec_transition_shims_np not available\n";
    return EXIT_FAILURE;
  }

  ret = posix_spawnattr_enable_memory_tagging_fn(&attr, /*unused=*/0);
  if (ret != 0) {
    WithColor::error()
        << "posix_spawnattr_set_use_sec_transition_shims_np failed\n";
    return EXIT_FAILURE;
  }

  std::vector<const char *> new_envp = get_extended_env(envp);

  pid_t pid;
  ret = posix_spawn(&pid, program, /*file_actions=*/nullptr, &attr,
                    const_cast<char **>(new_args),
                    const_cast<char **>(new_envp.data()));
  if (ret != 0) {
    WithColor::error() << "posix_spawn failed with error " << ret << "("
                       << strerror(ret) << ")\n";
    return EXIT_FAILURE;
  }

  int status;
  if (waitpid(pid, &status, 0) == -1) {
    WithColor::error() << "waitpid failed\n";
    return EXIT_FAILURE;
  }

  return WEXITSTATUS(status);
}
