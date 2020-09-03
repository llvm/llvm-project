"""GDB pretty printers for MLIR types."""

import gdb.printing

class IdentifierPrinter:
  """Prints an mlir::Identifier instance."""

  def __init__(self, val):
    self.entry = val['entry']

  def to_string(self):
    ptr = (self.entry + 1).cast(gdb.lookup_type('char').pointer());
    return ptr.string(length=self.entry['keyLength'])

  def display_hint(self):
    return 'string'

pp = gdb.printing.RegexpCollectionPrettyPrinter('MLIRSupport')

pp.add_printer('mlir::Identifier', '^mlir::Identifier$', IdentifierPrinter)

gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)
