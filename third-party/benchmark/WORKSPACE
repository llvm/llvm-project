workspace(name = "com_github_google_benchmark")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "com_google_absl",
    sha256 = "f41868f7a938605c92936230081175d1eae87f6ea2c248f41077c8f88316f111",
    strip_prefix = "abseil-cpp-20200225.2",
    urls = ["https://github.com/abseil/abseil-cpp/archive/20200225.2.tar.gz"],
)

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.11.0",
)

http_archive(
    name = "pybind11",
    build_file = "@//bindings/python:pybind11.BUILD",
    sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
    strip_prefix = "pybind11-2.4.3",
    urls = ["https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz"],
)

new_local_repository(
    name = "python_headers",
    build_file = "@//bindings/python:python_headers.BUILD",
    path = "/usr/include/python3.6",  # May be overwritten by setup.py.
)

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
)

load("@rules_python//python:pip.bzl", pip3_install="pip_install")

pip3_install(
   name = "py_deps",
   requirements = "//:requirements.txt",
)
