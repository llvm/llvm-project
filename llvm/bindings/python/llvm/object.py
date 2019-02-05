#===- object.py - Python Object Bindings --------------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

r"""
Object File Interface
=====================

This module provides an interface for reading information from object files
(e.g. binary executables and libraries).

Using this module, you can obtain information about an object file's sections,
symbols, and relocations. These are represented by the classes ObjectFile,
Section, Symbol, and Relocation, respectively.

Usage
-----

The only way to use this module is to start by creating an ObjectFile. You can
create an ObjectFile by loading a file (specified by its path) or by creating a
llvm.core.MemoryBuffer and loading that.

Once you have an object file, you can inspect its sections and symbols directly
by calling get_sections() and get_symbols() respectively. To inspect
relocations, call get_relocations() on a Section instance.

Iterator Interface
------------------

The LLVM bindings expose iteration over sections, symbols, and relocations in a
way that only allows one instance to be operated on at a single time. This is
slightly annoying from a Python perspective, as it isn't very Pythonic to have
objects that "expire" but are still active from a dynamic language.

To aid working around this limitation, each Section, Symbol, and Relocation
instance caches its properties after first access. So, if the underlying
iterator is advanced, the properties can still be obtained provided they have
already been retrieved.

In addition, we also provide a "cache" method on each class to cache all
available data. You can call this on each obtained instance. Or, you can pass
cache=True to the appropriate get_XXX() method to have this done for you.

Here are some examples on how to perform iteration:

    obj = ObjectFile(filename='/bin/ls')

    # This is OK. Each Section is only accessed inside its own iteration slot.
    section_names = []
    for section in obj.get_sections():
        section_names.append(section.name)

    # This is NOT OK. You perform a lookup after the object has expired.
    symbols = list(obj.get_symbols())
    for symbol in symbols:
        print symbol.name # This raises because the object has expired.

    # In this example, we mix a working and failing scenario.
    symbols = []
    for symbol in obj.get_symbols():
        symbols.append(symbol)
        print symbol.name

    for symbol in symbols:
        print symbol.name # OK
        print symbol.address # NOT OK. We didn't look up this property before.

    # Cache everything up front.
    symbols = list(obj.get_symbols(cache=True))
    for symbol in symbols:
        print symbol.name # OK

"""

from ctypes import c_char_p
from ctypes import c_char
from ctypes import POINTER
from ctypes import c_uint64
from ctypes import string_at

from .common import CachedProperty
from .common import LLVMObject
from .common import c_object_p
from .common import get_library
from .core import MemoryBuffer

__all__ = [
    "lib",
    "ObjectFile",
    "Relocation",
    "Section",
    "Symbol",
]

class ObjectFile(LLVMObject):
    """Represents an object/binary file."""

    def __init__(self, filename=None, contents=None):
        """Construct an instance from a filename or binary data.

        filename must be a path to a file that can be opened with open().
        contents can be either a native Python buffer type (like str) or a
        llvm.core.MemoryBuffer instance.
        """
        if contents:
            assert isinstance(contents, MemoryBuffer)

        if filename is not None:
            contents = MemoryBuffer(filename=filename)

        if contents is None:
            raise Exception('No input found.')

        ptr = lib.LLVMCreateObjectFile(contents)
        LLVMObject.__init__(self, ptr, disposer=lib.LLVMDisposeObjectFile)
        self.take_ownership(contents)

    def get_sections(self, cache=False):
        """Obtain the sections in this object file.

        This is a generator for llvm.object.Section instances.

        Sections are exposed as limited-use objects. See the module's
        documentation on iterators for more.
        """
        sections = lib.LLVMGetSections(self)
        last = None
        while True:
            if lib.LLVMIsSectionIteratorAtEnd(self, sections):
                break

            last = Section(sections)
            if cache:
                last.cache()

            yield last

            lib.LLVMMoveToNextSection(sections)
            last.expire()

        if last is not None:
            last.expire()

        lib.LLVMDisposeSectionIterator(sections)

    def get_symbols(self, cache=False):
        """Obtain the symbols in this object file.

        This is a generator for llvm.object.Symbol instances.

        Each Symbol instance is a limited-use object. See this module's
        documentation on iterators for more.
        """
        symbols = lib.LLVMGetSymbols(self)
        last = None
        while True:
            if lib.LLVMIsSymbolIteratorAtEnd(self, symbols):
                break

            last = Symbol(symbols, self)
            if cache:
                last.cache()

            yield last

            lib.LLVMMoveToNextSymbol(symbols)
            last.expire()

        if last is not None:
            last.expire()

        lib.LLVMDisposeSymbolIterator(symbols)

class Section(LLVMObject):
    """Represents a section in an object file."""

    def __init__(self, ptr):
        """Construct a new section instance.

        Section instances can currently only be created from an ObjectFile
        instance. Therefore, this constructor should not be used outside of
        this module.
        """
        LLVMObject.__init__(self, ptr)

        self.expired = False

    @CachedProperty
    def name(self):
        """Obtain the string name of the section.

        This is typically something like '.dynsym' or '.rodata'.
        """
        if self.expired:
            raise Exception('Section instance has expired.')

        return lib.LLVMGetSectionName(self)

    @CachedProperty
    def size(self):
        """The size of the section, in long bytes."""
        if self.expired:
            raise Exception('Section instance has expired.')

        return lib.LLVMGetSectionSize(self)

    @CachedProperty
    def contents(self):
        if self.expired:
            raise Exception('Section instance has expired.')

        siz = self.size

        r = lib.LLVMGetSectionContents(self)
        if r:
            return string_at(r, siz)
        return None

    @CachedProperty
    def address(self):
        """The address of this section, in long bytes."""
        if self.expired:
            raise Exception('Section instance has expired.')

        return lib.LLVMGetSectionAddress(self)

    def has_symbol(self, symbol):
        """Returns whether a Symbol instance is present in this Section."""
        if self.expired:
            raise Exception('Section instance has expired.')

        assert isinstance(symbol, Symbol)
        return lib.LLVMGetSectionContainsSymbol(self, symbol)

    def get_relocations(self, cache=False):
        """Obtain the relocations in this Section.

        This is a generator for llvm.object.Relocation instances.

        Each instance is a limited used object. See this module's documentation
        on iterators for more.
        """
        if self.expired:
            raise Exception('Section instance has expired.')

        relocations = lib.LLVMGetRelocations(self)
        last = None
        while True:
            if lib.LLVMIsRelocationIteratorAtEnd(self, relocations):
                break

            last = Relocation(relocations)
            if cache:
                last.cache()

            yield last

            lib.LLVMMoveToNextRelocation(relocations)
            last.expire()

        if last is not None:
            last.expire()

        lib.LLVMDisposeRelocationIterator(relocations)

    def cache(self):
        """Cache properties of this Section.

        This can be called as a workaround to the single active Section
        limitation. When called, the properties of the Section are fetched so
        they are still available after the Section has been marked inactive.
        """
        getattr(self, 'name')
        getattr(self, 'size')
        getattr(self, 'contents')
        getattr(self, 'address')

    def expire(self):
        """Expire the section.

        This is called internally by the section iterator.
        """
        self.expired = True

class Symbol(LLVMObject):
    """Represents a symbol in an object file."""
    def __init__(self, ptr, object_file):
        assert isinstance(ptr, c_object_p)
        assert isinstance(object_file, ObjectFile)

        LLVMObject.__init__(self, ptr)

        self.expired = False
        self._object_file = object_file

    @CachedProperty
    def name(self):
        """The str name of the symbol.

        This is often a function or variable name. Keep in mind that name
        mangling could be in effect.
        """
        if self.expired:
            raise Exception('Symbol instance has expired.')

        return lib.LLVMGetSymbolName(self)

    @CachedProperty
    def address(self):
        """The address of this symbol, in long bytes."""
        if self.expired:
            raise Exception('Symbol instance has expired.')

        return lib.LLVMGetSymbolAddress(self)

    @CachedProperty
    def size(self):
        """The size of the symbol, in long bytes."""
        if self.expired:
            raise Exception('Symbol instance has expired.')

        return lib.LLVMGetSymbolSize(self)

    @CachedProperty
    def section(self):
        """The Section to which this Symbol belongs.

        The returned Section instance does not expire, unlike Sections that are
        commonly obtained through iteration.

        Because this obtains a new section iterator each time it is accessed,
        calling this on a number of Symbol instances could be expensive.
        """
        sections = lib.LLVMGetSections(self._object_file)
        lib.LLVMMoveToContainingSection(sections, self)

        return Section(sections)

    def cache(self):
        """Cache all cacheable properties."""
        getattr(self, 'name')
        getattr(self, 'address')
        getattr(self, 'size')

    def expire(self):
        """Mark the object as expired to prevent future API accesses.

        This is called internally by this module and it is unlikely that
        external callers have a legitimate reason for using it.
        """
        self.expired = True

class Relocation(LLVMObject):
    """Represents a relocation definition."""
    def __init__(self, ptr):
        """Create a new relocation instance.

        Relocations are created from objects derived from Section instances.
        Therefore, this constructor should not be called outside of this
        module. See Section.get_relocations() for the proper method to obtain
        a Relocation instance.
        """
        assert isinstance(ptr, c_object_p)

        LLVMObject.__init__(self, ptr)

        self.expired = False

    @CachedProperty
    def offset(self):
        """The offset of this relocation, in long bytes."""
        if self.expired:
            raise Exception('Relocation instance has expired.')

        return lib.LLVMGetRelocationOffset(self)

    @CachedProperty
    def symbol(self):
        """The Symbol corresponding to this Relocation."""
        if self.expired:
            raise Exception('Relocation instance has expired.')

        ptr = lib.LLVMGetRelocationSymbol(self)
        return Symbol(ptr)

    @CachedProperty
    def type_number(self):
        """The relocation type, as a long."""
        if self.expired:
            raise Exception('Relocation instance has expired.')

        return lib.LLVMGetRelocationType(self)

    @CachedProperty
    def type_name(self):
        """The relocation type's name, as a str."""
        if self.expired:
            raise Exception('Relocation instance has expired.')

        return lib.LLVMGetRelocationTypeName(self)

    @CachedProperty
    def value_string(self):
        if self.expired:
            raise Exception('Relocation instance has expired.')

        return lib.LLVMGetRelocationValueString(self)

    def expire(self):
        """Expire this instance, making future API accesses fail."""
        self.expired = True

    def cache(self):
        """Cache all cacheable properties on this instance."""
        getattr(self, 'address')
        getattr(self, 'offset')
        getattr(self, 'symbol')
        getattr(self, 'type')
        getattr(self, 'type_name')
        getattr(self, 'value_string')

def register_library(library):
    """Register function prototypes with LLVM library instance."""

    # Object.h functions
    library.LLVMCreateObjectFile.argtypes = [MemoryBuffer]
    library.LLVMCreateObjectFile.restype = c_object_p

    library.LLVMDisposeObjectFile.argtypes = [ObjectFile]

    library.LLVMGetSections.argtypes = [ObjectFile]
    library.LLVMGetSections.restype = c_object_p

    library.LLVMDisposeSectionIterator.argtypes = [c_object_p]

    library.LLVMIsSectionIteratorAtEnd.argtypes = [ObjectFile, c_object_p]
    library.LLVMIsSectionIteratorAtEnd.restype = bool

    library.LLVMMoveToNextSection.argtypes = [c_object_p]

    library.LLVMMoveToContainingSection.argtypes = [c_object_p, c_object_p]

    library.LLVMGetSymbols.argtypes = [ObjectFile]
    library.LLVMGetSymbols.restype = c_object_p

    library.LLVMDisposeSymbolIterator.argtypes = [c_object_p]

    library.LLVMIsSymbolIteratorAtEnd.argtypes = [ObjectFile, c_object_p]
    library.LLVMIsSymbolIteratorAtEnd.restype = bool

    library.LLVMMoveToNextSymbol.argtypes = [c_object_p]

    library.LLVMGetSectionName.argtypes = [c_object_p]
    library.LLVMGetSectionName.restype = c_char_p

    library.LLVMGetSectionSize.argtypes = [c_object_p]
    library.LLVMGetSectionSize.restype = c_uint64

    library.LLVMGetSectionContents.argtypes = [c_object_p]
    # Can't use c_char_p here as it isn't a NUL-terminated string.
    library.LLVMGetSectionContents.restype = POINTER(c_char)

    library.LLVMGetSectionAddress.argtypes = [c_object_p]
    library.LLVMGetSectionAddress.restype = c_uint64

    library.LLVMGetSectionContainsSymbol.argtypes = [c_object_p, c_object_p]
    library.LLVMGetSectionContainsSymbol.restype = bool

    library.LLVMGetRelocations.argtypes = [c_object_p]
    library.LLVMGetRelocations.restype = c_object_p

    library.LLVMDisposeRelocationIterator.argtypes = [c_object_p]

    library.LLVMIsRelocationIteratorAtEnd.argtypes = [c_object_p, c_object_p]
    library.LLVMIsRelocationIteratorAtEnd.restype = bool

    library.LLVMMoveToNextRelocation.argtypes = [c_object_p]

    library.LLVMGetSymbolName.argtypes = [Symbol]
    library.LLVMGetSymbolName.restype = c_char_p

    library.LLVMGetSymbolAddress.argtypes = [Symbol]
    library.LLVMGetSymbolAddress.restype = c_uint64

    library.LLVMGetSymbolSize.argtypes = [Symbol]
    library.LLVMGetSymbolSize.restype = c_uint64

    library.LLVMGetRelocationOffset.argtypes = [c_object_p]
    library.LLVMGetRelocationOffset.restype = c_uint64

    library.LLVMGetRelocationSymbol.argtypes = [c_object_p]
    library.LLVMGetRelocationSymbol.restype = c_object_p

    library.LLVMGetRelocationType.argtypes = [c_object_p]
    library.LLVMGetRelocationType.restype = c_uint64

    library.LLVMGetRelocationTypeName.argtypes = [c_object_p]
    library.LLVMGetRelocationTypeName.restype = c_char_p

    library.LLVMGetRelocationValueString.argtypes = [c_object_p]
    library.LLVMGetRelocationValueString.restype = c_char_p

lib = get_library()
register_library(lib)
