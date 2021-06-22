# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository rules to configure the terminfo used by LLVM.

Most users should pick one of the explicit rules to configure their use of terminfo
with LLVM:
- `llvm_terminfo_system` will detect and link against a terminfo-implementing
  system library (non-hermetically).
- 'llvm_terminfo_disable` will disable terminfo completely.

If you would like to make your build configurable, you can use
`llvm_terminfo_from_env`. By default, this will disable terminfo, but will
inspect the environment variable (most easily set with a `--repo_env` flag to
the Bazel invocation) `BAZEL_LLVM_TERMINFO_STRATEGY`. If it is set to
`system` then it will behave the same as `llvm_terminfo_system`. Any other
setting will disable terminfo the same as not setting it at all.
"""

def _llvm_terminfo_disable_impl(repository_ctx):
    repository_ctx.template(
        "BUILD",
        repository_ctx.attr._disable_build_template,
        executable = False,
    )

_terminfo_disable_attrs = {
    "_disable_build_template": attr.label(
        default = Label("//deps_impl:terminfo_disable.BUILD"),
        allow_single_file = True,
    ),
}

llvm_terminfo_disable = repository_rule(
    implementation = _llvm_terminfo_disable_impl,
    attrs = _terminfo_disable_attrs,
)

def _find_c_compiler(repository_ctx):
    """Returns the path to a plausible C compiler.

    This routine will only reliably work on roughly POSIX-y systems as it
    ultimately falls back on the `cc` binary. Fortunately, the thing we are
    trying to use it for (detecting if a trivial source file can compile and
    link against a particular library) requires very little.
    """
    cc_env = repository_ctx.os.environ.get("CC")
    cc = None
    if cc_env:
        if "/" in cc_env:
            return repository_ctx.path(cc_env)
        else:
            return repository_ctx.which(cc_env)

    # Look for Clang, GCC, and the POSIX / UNIX specified C compiler
    # binaries.
    for compiler in ["clang", "gcc", "c99", "c89", "cc"]:
        cc = repository_ctx.which(compiler)
        if cc:
            return cc

    return None

def _try_link(repository_ctx, cc, source, linker_flags):
    """Returns `True` if able to link the source with the linker flag.

    Given a source file that contains references to library routines, this
    will check that when linked with the provided linker flag, those
    references are successfully resolved. This routine assumes a generally
    POSIX-y and GCC-ish compiler and environment and shouldn't be expected to
    work outside of that.
    """
    cmd = [
        cc,
        # Force discard the linked executable.
        "-o",
        "/dev/null",
        # Leave language detection to the compiler.
        source,
    ]

    # The linker flag must be valid for a compiler invocation of the link step,
    # so just append them to the command.
    cmd += linker_flags
    exec_result = repository_ctx.execute(cmd, timeout = 20)
    return exec_result.return_code == 0

def _llvm_terminfo_system_impl(repository_ctx):
    # LLVM doesn't need terminfo support on Windows, so just disable it.
    if repository_ctx.os.name.lower().find("windows") != -1:
        _llvm_terminfo_disable_impl(repository_ctx)
        return

    if len(repository_ctx.attr.system_linkopts) > 0:
        linkopts = repository_ctx.attr.system_linkopts
    else:
        required = repository_ctx.attr.system_required

        # Find a C compiler we can use to detect viable linkopts on this system.
        cc = _find_c_compiler(repository_ctx)
        if not cc:
            if required:
                fail("Failed  to find a C compiler executable")
            else:
                _llvm_terminfo_disable_impl(repository_ctx)
                return

        # Get the source file we use to detect successful linking of terminfo.
        source = repository_ctx.path(repository_ctx.attr._terminfo_test_source)

        # Collect the candidate linkopts and wrap them into a list. Ideally,
        # these would be provided as lists, but Bazel doesn't currently
        # support that. See: https://github.com/bazelbuild/bazel/issues/12178
        linkopts_candidates = [[x] for x in repository_ctx.attr.candidate_system_linkopts]

        # For each candidate, try to use it to link our test source file.
        for linkopts_candidate in linkopts_candidates:
            if _try_link(repository_ctx, cc, source, linkopts_candidate):
                linkopts = linkopts_candidate
                break

        # If we never found a viable linkopts candidate, either error or disable
        # terminfo for LLVM.
        if not linkopts:
            if required:
                fail("Failed to detect which linkopt would successfully provide the " +
                     "necessary terminfo functionality")
            else:
                _llvm_terminfo_disable_impl(repository_ctx)
                return

    repository_ctx.template(
        "BUILD",
        repository_ctx.attr._system_build_template,
        substitutions = {
            "{TERMINFO_LINKOPTS}": str(linkopts),
        },
        executable = False,
    )

def _merge_attrs(attrs_list):
    attrs = {}
    for input_attrs in attrs_list:
        attrs.update(input_attrs)
    return attrs

_terminfo_system_attrs = _merge_attrs([_terminfo_disable_attrs, {
    "_system_build_template": attr.label(
        default = Label("//deps_impl:terminfo_system.BUILD"),
        allow_single_file = True,
    ),
    "_terminfo_test_source": attr.label(
        default = Label("//deps_impl:terminfo_test.c"),
        allow_single_file = True,
    ),
    "candidate_system_linkopts": attr.string_list(
        default = [
            "-lterminfo",
            "-ltinfo",
            "-lcurses",
            "-lncurses",
            "-lncursesw",
        ],
        doc = "Candidate linkopts to test and see if they can link " +
              "successfully.",
    ),
    "system_required": attr.bool(
        default = False,
        doc = "Require that one of the candidates is detected successfully on POSIX platforms where it is needed.",
    ),
    "system_linkopts": attr.string_list(
        default = [],
        doc = "If non-empty, a specific array of linkopts to use to " +
              "successfully link against the terminfo library. No " +
              "detection is performed if this option is provided, it " +
              "directly forces the use of these link options. No test is " +
              "run to determine if they are valid or work correctly either.",
    ),
}])

llvm_terminfo_system = repository_rule(
    implementation = _llvm_terminfo_system_impl,
    configure = True,
    local = True,
    attrs = _terminfo_system_attrs,
)

def _llvm_terminfo_from_env_impl(repository_ctx):
    terminfo_strategy = repository_ctx.os.environ.get("BAZEL_LLVM_TERMINFO_STRATEGY")
    if terminfo_strategy == "system":
        _llvm_terminfo_system_impl(repository_ctx)
    else:
        _llvm_terminfo_disable_impl(repository_ctx)

llvm_terminfo_from_env = repository_rule(
    implementation = _llvm_terminfo_from_env_impl,
    configure = True,
    local = True,
    attrs = _merge_attrs([_terminfo_disable_attrs, _terminfo_system_attrs]),
    environ = ["BAZEL_LLVM_TERMINFO_STRATEGY", "CC"],
)
