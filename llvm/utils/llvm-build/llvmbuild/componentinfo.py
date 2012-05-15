"""
Descriptor objects for entities that are part of the LLVM project.
"""

import ConfigParser
import StringIO
import sys

from util import *

class ParseError(Exception):
    pass

class ComponentInfo(object):
    """
    Base class for component descriptions.
    """

    type_name = None

    @staticmethod
    def parse_items(items, has_dependencies = True):
        kwargs = {}
        kwargs['name'] = items.get_string('name')
        kwargs['parent'] = items.get_optional_string('parent')
        if has_dependencies:
            kwargs['dependencies'] = items.get_list('dependencies')
        return kwargs

    def __init__(self, subpath, name, dependencies, parent):
        if not subpath.startswith('/'):
            raise ValueError,"invalid subpath: %r" % subpath
        self.subpath = subpath
        self.name = name
        self.dependencies = list(dependencies)

        # The name of the parent component to logically group this component
        # under.
        self.parent = parent

        # The parent instance, once loaded.
        self.parent_instance = None
        self.children = []

        # The original source path.
        self._source_path = None

        # A flag to mark "special" components which have some amount of magic
        # handling (generally based on command line options).
        self._is_special_group = False

    def set_parent_instance(self, parent):
        assert parent.name == self.parent, "Unexpected parent!"
        self.parent_instance = parent
        self.parent_instance.children.append(self)

    def get_component_references(self):
        """get_component_references() -> iter

        Return an iterator over the named references to other components from
        this object. Items are of the form (reference-type, component-name).
        """

        # Parent references are handled specially.
        for r in self.dependencies:
            yield ('dependency', r)

    def get_llvmbuild_fragment(self):
        abstract

    def get_parent_target_group(self):
        """get_parent_target_group() -> ComponentInfo or None

        Return the nearest parent target group (if any), or None if the
        component is not part of any target group.
        """

        # If this is a target group, return it.
        if self.type_name == 'TargetGroup':
            return self

        # Otherwise recurse on the parent, if any.
        if self.parent_instance:
            return self.parent_instance.get_parent_target_group()

class GroupComponentInfo(ComponentInfo):
    """
    Group components have no semantics as far as the build system are concerned,
    but exist to help organize other components into a logical tree structure.
    """

    type_name = 'Group'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items, has_dependencies = False)
        return GroupComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, parent):
        ComponentInfo.__init__(self, subpath, name, [], parent)

    def get_llvmbuild_fragment(self):
        result = StringIO.StringIO()
        print >>result, 'type = %s' % self.type_name
        print >>result, 'name = %s' % self.name
        print >>result, 'parent = %s' % self.parent
        return result.getvalue()

class LibraryComponentInfo(ComponentInfo):
    type_name = 'Library'

    @staticmethod
    def parse_items(items):
        kwargs = ComponentInfo.parse_items(items)
        kwargs['library_name'] = items.get_optional_string('library_name')
        kwargs['required_libraries'] = items.get_list('required_libraries')
        kwargs['add_to_library_groups'] = items.get_list(
            'add_to_library_groups')
        kwargs['installed'] = items.get_optional_bool('installed', True)
        return kwargs

    @staticmethod
    def parse(subpath, items):
        kwargs = LibraryComponentInfo.parse_items(items)
        return LibraryComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, dependencies, parent, library_name,
                 required_libraries, add_to_library_groups, installed):
        ComponentInfo.__init__(self, subpath, name, dependencies, parent)

        # If given, the name to use for the library instead of deriving it from
        # the component name.
        self.library_name = library_name

        # The names of the library components which are required when linking
        # with this component.
        self.required_libraries = list(required_libraries)

        # The names of the library group components this component should be
        # considered part of.
        self.add_to_library_groups = list(add_to_library_groups)

        # Whether or not this library is installed.
        self.installed = installed

    def get_component_references(self):
        for r in ComponentInfo.get_component_references(self):
            yield r
        for r in self.required_libraries:
            yield ('required library', r)
        for r in self.add_to_library_groups:
            yield ('library group', r)

    def get_llvmbuild_fragment(self):
        result = StringIO.StringIO()
        print >>result, 'type = %s' % self.type_name
        print >>result, 'name = %s' % self.name
        print >>result, 'parent = %s' % self.parent
        if self.library_name is not None:
            print >>result, 'library_name = %s' % self.library_name
        if self.required_libraries:
            print >>result, 'required_libraries = %s' % ' '.join(
                self.required_libraries)
        if self.add_to_library_groups:
            print >>result, 'add_to_library_groups = %s' % ' '.join(
                self.add_to_library_groups)
        if not self.installed:
            print >>result, 'installed = 0'
        return result.getvalue()

    def get_library_name(self):
        return self.library_name or self.name

    def get_prefixed_library_name(self):
        """
        get_prefixed_library_name() -> str

        Return the library name prefixed by the project name. This is generally
        what the library name will be on disk.
        """

        basename = self.get_library_name()

        # FIXME: We need to get the prefix information from an explicit project
        # object, or something.
        if basename in ('gtest', 'gtest_main'):
            return basename

        return 'LLVM%s' % basename

    def get_llvmconfig_component_name(self):
        return self.get_library_name().lower()

class OptionalLibraryComponentInfo(LibraryComponentInfo):
    type_name = "OptionalLibrary"

    @staticmethod
    def parse(subpath, items):
      kwargs = LibraryComponentInfo.parse_items(items)
      return OptionalLibraryComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, dependencies, parent, library_name,
                 required_libraries, add_to_library_groups, installed):
      LibraryComponentInfo.__init__(self, subpath, name, dependencies, parent,
                                    library_name, required_libraries,
                                    add_to_library_groups, installed)

class LibraryGroupComponentInfo(ComponentInfo):
    type_name = 'LibraryGroup'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items, has_dependencies = False)
        kwargs['required_libraries'] = items.get_list('required_libraries')
        kwargs['add_to_library_groups'] = items.get_list(
            'add_to_library_groups')
        return LibraryGroupComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, parent, required_libraries = [],
                 add_to_library_groups = []):
        ComponentInfo.__init__(self, subpath, name, [], parent)

        # The names of the library components which are required when linking
        # with this component.
        self.required_libraries = list(required_libraries)

        # The names of the library group components this component should be
        # considered part of.
        self.add_to_library_groups = list(add_to_library_groups)

    def get_component_references(self):
        for r in ComponentInfo.get_component_references(self):
            yield r
        for r in self.required_libraries:
            yield ('required library', r)
        for r in self.add_to_library_groups:
            yield ('library group', r)

    def get_llvmbuild_fragment(self):
        result = StringIO.StringIO()
        print >>result, 'type = %s' % self.type_name
        print >>result, 'name = %s' % self.name
        print >>result, 'parent = %s' % self.parent
        if self.required_libraries and not self._is_special_group:
            print >>result, 'required_libraries = %s' % ' '.join(
                self.required_libraries)
        if self.add_to_library_groups:
            print >>result, 'add_to_library_groups = %s' % ' '.join(
                self.add_to_library_groups)
        return result.getvalue()

    def get_llvmconfig_component_name(self):
        return self.name.lower()

class TargetGroupComponentInfo(ComponentInfo):
    type_name = 'TargetGroup'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items, has_dependencies = False)
        kwargs['required_libraries'] = items.get_list('required_libraries')
        kwargs['add_to_library_groups'] = items.get_list(
            'add_to_library_groups')
        kwargs['has_jit'] = items.get_optional_bool('has_jit', False)
        kwargs['has_asmprinter'] = items.get_optional_bool('has_asmprinter',
                                                           False)
        kwargs['has_asmparser'] = items.get_optional_bool('has_asmparser',
                                                          False)
        kwargs['has_disassembler'] = items.get_optional_bool('has_disassembler',
                                                             False)
        return TargetGroupComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, parent, required_libraries = [],
                 add_to_library_groups = [], has_jit = False,
                 has_asmprinter = False, has_asmparser = False,
                 has_disassembler = False):
        ComponentInfo.__init__(self, subpath, name, [], parent)

        # The names of the library components which are required when linking
        # with this component.
        self.required_libraries = list(required_libraries)

        # The names of the library group components this component should be
        # considered part of.
        self.add_to_library_groups = list(add_to_library_groups)

        # Whether or not this target supports the JIT.
        self.has_jit = bool(has_jit)

        # Whether or not this target defines an assembly printer.
        self.has_asmprinter = bool(has_asmprinter)

        # Whether or not this target defines an assembly parser.
        self.has_asmparser = bool(has_asmparser)

        # Whether or not this target defines an disassembler.
        self.has_disassembler = bool(has_disassembler)

        # Whether or not this target is enabled. This is set in response to
        # configuration parameters.
        self.enabled = False

    def get_component_references(self):
        for r in ComponentInfo.get_component_references(self):
            yield r
        for r in self.required_libraries:
            yield ('required library', r)
        for r in self.add_to_library_groups:
            yield ('library group', r)

    def get_llvmbuild_fragment(self):
        result = StringIO.StringIO()
        print >>result, 'type = %s' % self.type_name
        print >>result, 'name = %s' % self.name
        print >>result, 'parent = %s' % self.parent
        if self.required_libraries:
            print >>result, 'required_libraries = %s' % ' '.join(
                self.required_libraries)
        if self.add_to_library_groups:
            print >>result, 'add_to_library_groups = %s' % ' '.join(
                self.add_to_library_groups)
        for bool_key in ('has_asmparser', 'has_asmprinter', 'has_disassembler',
                         'has_jit'):
            if getattr(self, bool_key):
                print >>result, '%s = 1' % (bool_key,)
        return result.getvalue()

    def get_llvmconfig_component_name(self):
        return self.name.lower()

class ToolComponentInfo(ComponentInfo):
    type_name = 'Tool'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items)
        kwargs['required_libraries'] = items.get_list('required_libraries')
        return ToolComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, dependencies, parent,
                 required_libraries):
        ComponentInfo.__init__(self, subpath, name, dependencies, parent)

        # The names of the library components which are required to link this
        # tool.
        self.required_libraries = list(required_libraries)

    def get_component_references(self):
        for r in ComponentInfo.get_component_references(self):
            yield r
        for r in self.required_libraries:
            yield ('required library', r)

    def get_llvmbuild_fragment(self):
        result = StringIO.StringIO()
        print >>result, 'type = %s' % self.type_name
        print >>result, 'name = %s' % self.name
        print >>result, 'parent = %s' % self.parent
        print >>result, 'required_libraries = %s' % ' '.join(
            self.required_libraries)
        return result.getvalue()

class BuildToolComponentInfo(ToolComponentInfo):
    type_name = 'BuildTool'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items)
        kwargs['required_libraries'] = items.get_list('required_libraries')
        return BuildToolComponentInfo(subpath, **kwargs)

###

class IniFormatParser(dict):
    def get_list(self, key):
        # Check if the value is defined.
        value = self.get(key)
        if value is None:
            return []

        # Lists are just whitespace separated strings.
        return value.split()

    def get_optional_string(self, key):
        value = self.get_list(key)
        if not value:
            return None
        if len(value) > 1:
            raise ParseError("multiple values for scalar key: %r" % key)
        return value[0]

    def get_string(self, key):
        value = self.get_optional_string(key)
        if not value:
            raise ParseError("missing value for required string: %r" % key)
        return value

    def get_optional_bool(self, key, default = None):
        value = self.get_optional_string(key)
        if not value:
            return default
        if value not in ('0', '1'):
            raise ParseError("invalid value(%r) for boolean property: %r" % (
                    value, key))
        return bool(int(value))

    def get_bool(self, key):
        value = self.get_optional_bool(key)
        if value is None:
            raise ParseError("missing value for required boolean: %r" % key)
        return value

_component_type_map = dict(
    (t.type_name, t)
    for t in (GroupComponentInfo,
              LibraryComponentInfo, LibraryGroupComponentInfo,
              ToolComponentInfo, BuildToolComponentInfo,
              TargetGroupComponentInfo, OptionalLibraryComponentInfo))
def load_from_path(path, subpath):
    # Load the LLVMBuild.txt file as an .ini format file.
    parser = ConfigParser.RawConfigParser()
    parser.read(path)

    # Extract the common section.
    if parser.has_section("common"):
        common = IniFormatParser(parser.items("common"))
        parser.remove_section("common")
    else:
        common = IniFormatParser({})

    return common, _read_components_from_parser(parser, path, subpath)

def _read_components_from_parser(parser, path, subpath):
    # We load each section which starts with 'component' as a distinct component
    # description (so multiple components can be described in one file).
    for section in parser.sections():
        if not section.startswith('component'):
            # We don't expect arbitrary sections currently, warn the user.
            warning("ignoring unknown section %r in %r" % (section, path))
            continue

        # Determine the type of the component to instantiate.
        if not parser.has_option(section, 'type'):
            fatal("invalid component %r in %r: %s" % (
                    section, path, "no component type"))

        type_name = parser.get(section, 'type')
        type_class = _component_type_map.get(type_name)
        if type_class is None:
            fatal("invalid component %r in %r: %s" % (
                    section, path, "invalid component type: %r" % type_name))

        # Instantiate the component based on the remaining values.
        try:
            info = type_class.parse(subpath,
                                    IniFormatParser(parser.items(section)))
        except TypeError:
            print >>sys.stderr, "error: invalid component %r in %r: %s" % (
                section, path, "unable to instantiate: %r" % type_name)
            import traceback
            traceback.print_exc()
            raise SystemExit, 1
        except ParseError,e:
            fatal("unable to load component %r in %r: %s" % (
                    section, path, e.message))

        info._source_path = path
        yield info
