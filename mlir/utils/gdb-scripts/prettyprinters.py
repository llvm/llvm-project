"""GDB pretty printers for MLIR types."""

import gdb.printing


class IdentifierPrinter:
  """Prints an mlir::Identifier instance."""

  def __init__(self, val):
    self.entry = val['entry']

  def to_string(self):
    ptr = (self.entry + 1).cast(gdb.lookup_type('char').pointer())
    return ptr.string(length=self.entry['keyLength'])

  def display_hint(self):
    return 'string'


class StoragePrinter:
  """Prints bases of a struct and its fields."""

  def __init__(self, val):
    self.val = val

  def children(self):
    for field in self.val.type.fields():
      if field.is_base_class:
        yield ('<%s>' % field.name, self.val.cast(field.type))
      else:
        yield (field.name, self.val[field.name])


class TupleTypeStoragePrinter(StoragePrinter):

  def children(self):
    for child in StoragePrinter.children(self):
      yield child
    pointer_type = gdb.lookup_type('mlir::Type').pointer()
    elements = (self.val.address + 1).cast(pointer_type)
    for i in range(self.val['numElements']):
      yield 'elements[%u]' % i, elements[i]


class RankedTypeStoragePrinter(StoragePrinter):

  def children(self):
    for child in StoragePrinter.children(self):
      yield child
    for i in range(self.val['shapeSize']):
      yield 'shapeElements[%u]' % i, self.val['shapeElements'][i]


class MemRefTypeStoragePrinter(RankedTypeStoragePrinter):

  def children(self):
    for child in RankedTypeStoragePrinter.children(self):
      yield child
    for i in range(self.val['numAffineMaps']):
      yield 'affineMapsList[%u]' % i, self.val['affineMapsList'][i]


class FusedLocationStoragePrinter(StoragePrinter):

  def children(self):
    for child in StoragePrinter.children(self):
      yield child
    pointer_type = gdb.lookup_type('mlir::Location').pointer()
    elements = (self.val.address + 1).cast(pointer_type)
    for i in range(self.val['numLocs']):
      yield 'locs[%u]' % i, elements[i]


class StorageUserBasePrinter:
  """Printer for an mlir::detail::StorageUserBase instance."""

  def __init__(self, val):
    self.val = val

  def children(self):
    storage_type = self.val.type.template_argument(2)
    yield 'impl', self.val['impl'].dereference().cast(storage_type)


class StorageTypeMap:
  """Maps a TypeID to the corresponding type derived from StorageUserBase.

  Types need to be registered by name before the first lookup.
  """

  def __init__(self):
    self.map = None
    self.type_names = []

  def register_type(self, type_name):
    assert not self.map, 'register_type called after __getitem__'
    self.type_names += [type_name]

  def _init_map(self):
    """Lazy initialization  of self.map."""
    if self.map:
      return
    self.map = {}
    for type_name in self.type_names:
      concrete_type = gdb.lookup_type(type_name)
      try:
        storage = gdb.parse_and_eval(
            "&'mlir::detail::TypeIDExported::get<%s>()::instance'" % type_name)
      except gdb.error:
        # Skip when TypeID instance cannot be found in current context.
        continue
      if concrete_type and storage:
        self.map[int(storage)] = concrete_type

  def __getitem__(self, type_id):
    self._init_map()
    return self.map.get(int(type_id['storage']))


storage_type_map = StorageTypeMap()


def get_type_id_printer(val):
  """Returns a printer of the name of a mlir::TypeID."""

  class StringPrinter:

    def __init__(self, string):
      self.string = string

    def to_string(self):
      return self.string

  concrete_type = storage_type_map[val]
  if not concrete_type:
    return None
  return StringPrinter('"%s"' % concrete_type.name)


def get_attr_or_type_printer(val, get_type_id):
  """Returns a printer for mlir::Attribute or mlir::Type."""

  class UpcastPrinter:

    def __init__(self, val, type):
      self.val = val.cast(type)

    def children(self):
      yield 'cast<%s>' % self.val.type.name, self.val

  if not val['impl']:
    return None
  type_id = get_type_id(val['impl'].dereference())
  concrete_type = storage_type_map[type_id]
  if not concrete_type:
    return None
  return UpcastPrinter(val, concrete_type)


pp = gdb.printing.RegexpCollectionPrettyPrinter('MLIRSupport')

pp.add_printer('mlir::Identifier', '^mlir::Identifier$', IdentifierPrinter)

# Printers for types deriving from AttributeStorage or TypeStorage.
pp.add_printer('mlir::detail::FusedLocationStorage',
               '^mlir::detail::FusedLocationStorage',
               FusedLocationStoragePrinter)
pp.add_printer('mlir::detail::VectorTypeStorage',
               '^mlir::detail::VectorTypeStorage', RankedTypeStoragePrinter)
pp.add_printer('mlir::detail::RankedTensorTypeStorage',
               '^mlir::detail::RankedTensorTypeStorage',
               RankedTypeStoragePrinter)
pp.add_printer('mlir::detail::MemRefTypeStorage',
               '^mlir::detail::MemRefTypeStorage$', MemRefTypeStoragePrinter)
pp.add_printer('mlir::detail::TupleTypeStorage',
               '^mlir::detail::TupleTypeStorage$', TupleTypeStoragePrinter)

# Printers for Attribute::AttrBase or Type::TypeBase typedefs.
pp.add_printer('mlir::detail::StorageUserBase',
               '^mlir::detail::StorageUserBase<.*>$', StorageUserBasePrinter)

# Printers of types deriving from Attribute::AttrBase or Type::TypeBase.
for name in [
    # mlir/IR/Attributes.h
    'ArrayAttr',
    'DictionaryAttr',
    'FloatAttr',
    'IntegerAttr',
    'IntegerSetAttr',
    'OpaqueAttr',
    'StringAttr',
    'SymbolRefAttr',
    'TypeAttr',
    'UnitAttr',
    'DenseStringElementsAttr',
    'DenseIntOrFPElementsAttr',
    'OpaqueElementsAttr',
    'SparseElementsAttr',
    # mlir/IR/StandardTypes.h
    'ComplexType',
    'IndexType',
    'IntegerType',
    'Float16Type',
    'Float32Type',
    'Float64Type',
    'NoneType',
    'VectorType',
    'RankedTensorType',
    'UnrankedTensorType',
    'MemRefType',
    'UnrankedMemRefType',
    'TupleType',
    # mlir/IR/Location.h
    'CallSiteLoc',
    'FileLineColLoc',
    'FusedLoc',
    'NameLoc',
    'OpaqueLoc',
    'UnknownLoc'
]:
  storage_type_map.register_type('mlir::%s' % name)  # Register for upcasting.

pp.add_printer('mlir::TypeID', '^mlir::TypeID$', get_type_id_printer)


def add_attr_or_type_printers(name):
  """Adds printers for mlir::Attribute or mlir::Type and their Storage type."""
  get_type_id = lambda val: val['abstract%s' % name]['typeID']
  pp.add_printer('mlir::%s' % name, '^mlir::%s$' % name,
                 lambda val: get_attr_or_type_printer(val, get_type_id))
  pp.add_printer('mlir::%sStorage' % name, '^mlir::%sStorage$' % name,
                 lambda val: get_type_id_printer(get_type_id(val)))


# Upcasting printers of mlir::Attribute and mlir::Type.
for name in ['Attribute', 'Type']:
  add_attr_or_type_printers(name)

gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)
