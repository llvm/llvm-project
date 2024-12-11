cc_library(
    name = "pybind11",
    hdrs = glob(
        include = ["include/pybind11/**/*.h"],
        exclude = [
            # Deprecated file that just emits a warning
            "include/pybind11/common.h",
        ],
    ),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)
