def _workspace_root_impl(ctx):
    """Dynamically determine the workspace root from the current context.

    The path is made available as a `WORKSPACE_ROOT` environmment variable and
    may for instance be consumed in the `toolchains` attributes for `cc_library`
    and `genrule` targets.
    """
    return [
        platform_common.TemplateVariableInfo({
            "WORKSPACE_ROOT": ctx.label.workspace_root,
        }),
    ]

workspace_root = rule(
    implementation = _workspace_root_impl,
    attrs = {},
)
