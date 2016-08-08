import gdb.printing
class SmallStringPrinter:
  """Print an llvm::SmallString object."""

  def __init__(self, val):
    self.val = val

  def to_string(self):
    begin = self.val['BeginX']
    end = self.val['EndX']
    return begin.cast(gdb.lookup_type("char").pointer()).string(length = end - begin)

  def display_hint (self):
    return 'string'

class StringRefPrinter:
  """Print an llvm::StringRef object."""

  def __init__(self, val):
    self.val = val

  def to_string(self):
    return self.val['Data'].string(length =  self.val['Length'])

  def display_hint (self):
    return 'string'

class SmallVectorPrinter:
  """Print an llvm::SmallVector object."""

  class _iterator:
    def __init__(self, begin, end):
      self.cur = begin
      self.end = end
      self.count = 0

    def __iter__(self):
      return self

    def next(self):
      if self.cur == self.end:
        raise StopIteration
      count = self.count
      self.count = self.count + 1
      cur = self.cur
      self.cur = self.cur + 1
      return '[%d]' % count, cur.dereference()

    __next__ = next

  def __init__(self, val):
    self.val = val

  def children(self):
    t = self.val.type.template_argument(0).pointer()
    begin = self.val['BeginX'].cast(t)
    end = self.val['EndX'].cast(t)
    return self._iterator(begin, end)

  def to_string(self):
    t = self.val.type.template_argument(0).pointer()
    begin = self.val['BeginX'].cast(t)
    end = self.val['EndX'].cast(t)
    capacity = self.val['CapacityX'].cast(t)
    return 'llvm::SmallVector of length %d, capacity %d' % (end - begin, capacity - begin)

  def display_hint (self):
    return 'array'

class ArrayRefPrinter:
  """Print an llvm::ArrayRef object."""

  class _iterator:
    def __init__(self, begin, end):
      self.cur = begin
      self.end = end
      self.count = 0

    def __iter__(self):
      return self

    def next(self):
      if self.cur == self.end:
        raise StopIteration
      count = self.count
      self.count = self.count + 1
      cur = self.cur
      self.cur = self.cur + 1
      return '[%d]' % count, cur.dereference()

    __next__ = next

  def __init__(self, val):
    self.val = val

  def children(self):
    data = self.val['Data']
    return self._iterator(data, data + self.val['Length'])

  def to_string(self):
    return 'llvm::ArrayRef of length %d' % (self.val['Length'])

  def display_hint (self):
    return 'array'

pp = gdb.printing.RegexpCollectionPrettyPrinter("LLVMSupport")
pp.add_printer('llvm::SmallString', '^llvm::SmallString<.*>$', SmallStringPrinter)
pp.add_printer('llvm::StringRef', '^llvm::StringRef$', StringRefPrinter)
pp.add_printer('llvm::SmallVectorImpl', '^llvm::SmallVector(Impl)?<.*>$', SmallVectorPrinter)
pp.add_printer('llvm::ArrayRef', '^llvm::(Const)?ArrayRef<.*>$', ArrayRefPrinter)
gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)
