from .base import TestBase
from ..object import ObjectFile
from ..object import Relocation
from ..object import Section
from ..object import Symbol

class TestObjectFile(TestBase):
    def get_object_file(self):
        source = self.get_test_binary()
        return ObjectFile(filename=source)

    def test_create_from_file(self):
        self.get_object_file()

    def test_get_sections(self):
        o = self.get_object_file()

        count = 0
        for section in o.get_sections():
            count += 1
            assert isinstance(section, Section)
            assert isinstance(section.name, str)
            assert isinstance(section.size, long)
            assert isinstance(section.contents, str)
            assert isinstance(section.address, long)

        self.assertGreater(count, 0)

        for section in o.get_sections():
            section.cache()

    def test_get_symbols(self):
        o = self.get_object_file()

        count = 0
        for symbol in o.get_symbols():
            count += 1
            assert isinstance(symbol, Symbol)
            assert isinstance(symbol.name, str)
            assert isinstance(symbol.address, long)
            assert isinstance(symbol.size, long)
            assert isinstance(symbol.file_offset, long)

        self.assertGreater(count, 0)

        for symbol in o.get_symbols():
            symbol.cache()

    def test_symbol_section_accessor(self):
        o = self.get_object_file()

        for symbol in o.get_symbols():
            section = symbol.section
            assert isinstance(section, Section)

            break

    def test_get_relocations(self):
        o = self.get_object_file()
        for section in o.get_sections():
            for relocation in section.get_relocations():
                assert isinstance(relocation, Relocation)
                assert isinstance(relocation.address, long)
                assert isinstance(relocation.offset, long)
                assert isinstance(relocation.type_number, long)
                assert isinstance(relocation.type_name, str)
                assert isinstance(relocation.value_string, str)
