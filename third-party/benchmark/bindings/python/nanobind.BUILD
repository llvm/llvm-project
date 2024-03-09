load("@bazel_skylib//lib:selects.bzl", "selects")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "msvc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
)

selects.config_setting_group(
    name = "winplusmsvc",
    match_all = [
        "@platforms//os:windows",
        ":msvc_compiler",
    ],
)

cc_library(
    name = "nanobind",
    srcs = glob([
        "src/*.cpp",
    ]),
    additional_linker_inputs = select({
        "@platforms//os:macos": [":cmake/darwin-ld-cpython.sym"],
        "//conditions:default": [],
    }),
    copts = select({
        ":msvc_compiler": [
            "/EHsc",  # exceptions
            "/Os",  # size optimizations
            "/GL",  # LTO / whole program optimization
        ],
        # these should work on both clang and gcc.
        "//conditions:default": [
            "-fexceptions",
            "-flto",
            "-Os",
        ],
    }),
    includes = [
        "ext/robin_map/include",
        "include",
    ],
    linkopts = select({
        ":winplusmsvc": ["/LTGC"],  # Windows + MSVC.
        "@platforms//os:macos": ["-Wl,@$(location :cmake/darwin-ld-cpython.sym)"],  # Apple.
        "//conditions:default": [],
    }),
    textual_hdrs = glob(
        [
            "include/**/*.h",
            "src/*.h",
            "ext/robin_map/include/tsl/*.h",
        ],
    ),
    deps = ["@python_headers"],
)
