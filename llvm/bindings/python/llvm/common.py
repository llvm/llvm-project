#===- common.py - Python LLVM Bindings -----------------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

from ctypes import POINTER
from ctypes import c_void_p
from ctypes import cdll

import ctypes.util
import platform

# LLVM_VERSION: sync with PACKAGE_VERSION in autoconf/configure.ac and CMakeLists.txt
#               but leave out the 'svn' suffix.
LLVM_VERSION = '3.4'

__all__ = [
    'c_object_p',
    'get_library',
]

c_object_p = POINTER(c_void_p)

class LLVMObject(object):
    """Base class for objects that are backed by an LLVM data structure.

    This class should never be instantiated outside of this package.
    """
    def __init__(self, ptr, ownable=True, disposer=None):
        assert isinstance(ptr, c_object_p)

        self._ptr = self._as_parameter_ = ptr

        self._self_owned = True
        self._ownable = ownable
        self._disposer = disposer

        self._owned_objects = []

    def take_ownership(self, obj):
        """Take ownership of another object.

        When you take ownership of another object, you are responsible for
        destroying that object. In addition, a reference to that object is
        placed inside this object so the Python garbage collector will not
        collect the object while it is still alive in libLLVM.

        This method should likely only be called from within modules inside
        this package.
        """
        assert isinstance(obj, LLVMObject)

        self._owned_objects.append(obj)
        obj._self_owned = False

    def from_param(self):
        """ctypes function that converts this object to a function parameter."""
        return self._as_parameter_

    def __del__(self):
        if not hasattr(self, '_self_owned') or not hasattr(self, '_disposer'):
            return

        if self._self_owned and self._disposer:
            self._disposer(self)

class CachedProperty(object):
    """Decorator that caches the result of a property lookup.

    This is a useful replacement for @property. It is recommended to use this
    decorator on properties that invoke C API calls for which the result of the
    call will be idempotent.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        try:
            self.__doc__ = wrapped.__doc__
        except: # pragma: no cover
            pass

    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self

        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)

        return value

def get_library():
    """Obtain a reference to the llvm library."""

    # On Linux, ctypes.cdll.LoadLibrary() respects LD_LIBRARY_PATH
    # while ctypes.util.find_library() doesn't.
    # See http://docs.python.org/2/library/ctypes.html#finding-shared-libraries
    #
    # To make it possible to run the unit tests without installing the LLVM shared
    # library into a default linker search path.  Always Try ctypes.cdll.LoadLibrary()
    # with all possible library names first, then try ctypes.util.find_library().

    names = ['LLVM-' + LLVM_VERSION, 'LLVM-' + LLVM_VERSION + 'svn']
    t = platform.system()
    if t == 'Darwin':
        pfx, ext = 'lib', '.dylib'
    elif t == 'Windows':
        pfx, ext = '', '.dll'
    else:
        pfx, ext = 'lib', '.so'

    for i in names:
        try:
            lib = cdll.LoadLibrary(pfx + i + ext)
        except OSError:
            pass
        else:
            return lib

    for i in names:
        t = ctypes.util.find_library(i)
        if t:
            return cdll.LoadLibrary(t)
    raise Exception('LLVM shared library not found!')
