# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helper functions to resolve CMake templates."""

load("@bazel_skylib//lib:selects.bzl", "selects")

def cmakedefine_sset(define, value):
    """Translate `#cmakedefine DEFINE "${DEFINE}"` to `#define DEFINE "VALUE"`.

    Args:
        define: The string of a configurable define that may be set to a string
        value.

    Returns:
        A dict like
        `{'#cmakedefine DEFINE "${DEFINE}"': '#define DEFINE "VALUE"'}`.
    """
    return {
        '#cmakedefine {} "${{{}}}"'.format(
            define,
            define,
        ): '#define {} "{}"'.format(define, value),
    }

def cmakedefine_sunset(define):
    """Translate `#cmakedefine DEFINE "${DEFINE}"` to `/* #undef DEFINE */`.

    Args:
        define: The string of a configurable define that may be set to a string
        value.

    Returns:
        A dict like
        `{'#cmakedefine DEFINE "${DEFINE}"': "/* #undef DEFINE */"}`.
    """
    return {
        '#cmakedefine {} "${{{}}}"'.format(
            define,
            define,
        ): "/* #undef {} */".format(define),
    }

def cmakedefine_vset(define):
    """Translate `#cmakedefine DEFINE ${DEFINE}` to `#define DEFINE 1`.

    Args:
        define: The string of a configurable define that may be set to 1.

    Returns:
        A dict like `{"#cmakedefine DEFINE ${DEFINE}": "#define DEFINE 1"}`.
    """

    return {
        "#cmakedefine {} ${{{}}}".format(
            define,
            define,
        ): "#define {} 1".format(define),
    }

def cmakedefine_vunset(define):
    """Translate `#cmakedefine DEFINE ${DEFINE}` to `/* #undef DEFINE */`.

    Args:
        define: The string of a configurable define that may be set to a value.

    Returns:
        A dict like `{"#cmakedefine DEFINE ${DEFINE}": "/* #undef DEFINE */"}`.
    """
    return {
        "#cmakedefine {} ${{{}}}".format(
            define,
            define,
        ): "/* #undef {} */".format(define),
    }

def cmakedefine_set(define):
    """Translate `#cmakedefine DEFINE` to `#define DEFINE 1`.

    Args:
        define: The string of a configurable define that may be set or unset.

    Returns:
        A dict like `{"#cmakedefine DEFINE": "#define DEFINE 1"}`.
    """
    return {
        "#cmakedefine {}".format(
            define,
        ): "#define {} 1".format(define),
    }

def cmakedefine_unset(define):
    """Translate `#cmakedefine DEFINE` to `/* #undef DEFINE */`.

    Args:
        define: The string of a configurable define that may be set or unset.

    Returns:
        A dict like `{"#cmakedefine DEFINE": "/* #undef DEFINE */"}`.
    """
    return {
        "#cmakedefine {}".format(
            define,
        ): "/* #undef {} */".format(define),
    }

def cmakedefine(
        define,
        enable = "//conditions:default",
        disable = "//conditions:default"):
    """Translate `#cmakedefine DEFINE ${DEFINE}` to `#define DEFINE 1` or
    `/* #undef DEFINE */`.

    Args:
        define: The string of a configurable define that may be unset or set to
                a value.
        enable: A `Label` or tuple of `Labels` declaring conditions that set
            the define to 1. Defaults to `//conditions:default`.
        disable: A `Label` or tuple of `Labels` declaring conditions that leave
            the define undefined. Defaults to `//conditions:default`.

    Returns:
        A `selects.with_or` of the structure:

            selects.with_or({
                enable: {
                    "#cmakedefine DEFINE ${DEFINE}": "#define DEFINE 1",
                },
                disable: {
                    "#cmakedefine DEFINE ${DEFINE}": "/* #undef DEFINE */",
                },
            })

    Raises an error if `enable` and `disable` are both left at their defaults.
    """
    return selects.with_or({
        enable: cmakedefine_vset(define),
        disable: cmakedefine_vunset(define),
    })

def cmakedefine01_on(define):
    """Translate `#cmakedefine01 DEFINE` to `#define DEFINE 1`.

    Args:
        define: The string of a configurable define that may be 0 or 1.

    Returns:
        A dict like `{"#cmakedefine01 DEFINE": "#define DEFINE 1"}`.
    """
    return {
        "#cmakedefine01 {}".format(define): "#define {} 1".format(define),
    }

def cmakedefine01_off(define):
    """Translate `#cmakedefine01 DEFINE` to `#define DEFINE 0`.

    Args:
        define: The string of a configurable define that may be 0 or 1.

    Returns:
        A dict like `{"#cmakedefine01 DEFINE": "#define DEFINE 0"}`.
    """
    return {
        "#cmakedefine01 {}".format(define): "#define {} 0".format(define),
    }

def cmakedefine01(
        define,
        enable = "//conditions:default",
        disable = "//conditions:default"):
    """Translate `#cmakedefine01 DEFINE` to `#define DEFINE 1` or
    `#define DEFINE 0`.

    Args:
        define: The string of a configurable define that may be set to 0 or 1.
        enable: A `Label` or tuple of `Labels` declaring conditions that set
            the define to 1. Defaults to `//conditions:default`.
        disable: A `Label` or tuple of `Labels` declaring conditions that set
            the define to 0. Defaults to `//conditions:default`.

    Returns:
        A `selects.with_or` of the structure:

            selects.with_or({
                enable: {"#cmakedefine01 DEFINE": "#define DEFINE 1"},
                disable: {"#cmakedefine01 DEFINE": "#define DEFINE 0"},
            })

    Raises an error if `enable` and `disable` are both left at their defaults.
    """
    return selects.with_or({
        enable: cmakedefine01_on(define),
        disable: cmakedefine01_off(define),
    })
