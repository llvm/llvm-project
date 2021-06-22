# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository rules to configure the zlib used by LLVM.

Most users should pick one of the explicit rules to configure their use of zlib
with LLVM:
- `llvm_zlib_external` will link against an external Bazel zlib repository.
- `llvm_zlib_system` will link against the system zlib (non-hermetically).
- 'llvm_zlib_disable` will disable zlib completely.

If you would like to make your build configurable, you can use
`llvm_zlib_from_env`. By default, this will disable zlib, but will inspect
the environment variable (most easily set with a `--repo_env` flag to the
Bazel invocation) `BAZEL_LLVM_ZLIB_STRATEGY`. If it is set to `external`,
then it will behave the same as `llvm_zlib_external`. If it is set to
`system` then it will behave the same as `llvm_zlib_system`. Any other
setting will disable zlib the same as not setting it at all.
"""

def _llvm_zlib_external_impl(repository_ctx):
    repository_ctx.template(
        "BUILD",
        repository_ctx.attr._external_build_template,
        substitutions = {
            "@external_zlib_repo//:zlib_rule": str(repository_ctx.attr.external_zlib),
        },
        executable = False,
    )

llvm_zlib_external = repository_rule(
    implementation = _llvm_zlib_external_impl,
    attrs = {
        "_external_build_template": attr.label(
            default = Label("//deps_impl:zlib_external.BUILD"),
            allow_single_file = True,
        ),
        "external_zlib": attr.label(
            doc = "The dependency that should be used for the external zlib library.",
            mandatory = True,
        ),
    },
)

def _llvm_zlib_system_impl(repository_ctx):
    repository_ctx.template(
        "BUILD",
        repository_ctx.attr._system_build_template,
        executable = False,
    )

# While it may seem like this needs to be local, it doesn't actually inspect
# any local state, it just configures to build against that local state.
llvm_zlib_system = repository_rule(
    implementation = _llvm_zlib_system_impl,
    attrs = {
        "_system_build_template": attr.label(
            default = Label("//deps_impl:zlib_system.BUILD"),
            allow_single_file = True,
        ),
    },
)

def _llvm_zlib_disable_impl(repository_ctx):
    repository_ctx.template(
        "BUILD",
        repository_ctx.attr._disable_build_template,
        executable = False,
    )

llvm_zlib_disable = repository_rule(
    implementation = _llvm_zlib_disable_impl,
    attrs = {
        "_disable_build_template": attr.label(
            default = Label("//deps_impl:zlib_disable.BUILD"),
            allow_single_file = True,
        ),
    },
)

def _llvm_zlib_from_env_impl(repository_ctx):
    zlib_strategy = repository_ctx.os.environ.get("BAZEL_LLVM_ZLIB_STRATEGY")
    if zlib_strategy == "external":
        _llvm_zlib_external_impl(repository_ctx)
    elif zlib_strategy == "system":
        _llvm_zlib_system_impl(repository_ctx)
    else:
        _llvm_zlib_disable_impl(repository_ctx)

llvm_zlib_from_env = repository_rule(
    implementation = _llvm_zlib_from_env_impl,
    attrs = {
        "_disable_build_template": attr.label(
            default = Label("//deps_impl:zlib_disable.BUILD"),
            allow_single_file = True,
        ),
        "_external_build_template": attr.label(
            default = Label("//deps_impl:zlib_external.BUILD"),
            allow_single_file = True,
        ),
        "_system_build_template": attr.label(
            default = Label("//deps_impl:zlib_system.BUILD"),
            allow_single_file = True,
        ),
        "external_zlib": attr.label(
            doc = "The dependency that should be used for the external zlib library.",
            mandatory = True,
        ),
    },
    environ = ["BAZEL_LLVM_ZLIB_STRATEGY"],
)
