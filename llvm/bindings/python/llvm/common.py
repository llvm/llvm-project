#===- common.py - Python LLVM Bindings -----------------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

from ctypes import POINTER
from ctypes import c_void_p
from ctypes import cdll

import ctypes.util

__all__ = [
    'c_object_p',
    'find_library',
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

def find_library():
    # FIXME should probably have build system define absolute path of shared
    # library at install time.
    for lib in ['LLVM-3.1svn', 'libLLVM-3.1svn', 'LLVM', 'libLLVM']:
        result = ctypes.util.find_library(lib)
        if result:
            return result

    return None

def get_library():
    """Obtain a reference to the llvm library."""
    lib = find_library()
    if not lib:
        raise Exception('LLVM shared library not found!')

    return cdll.LoadLibrary(lib)
