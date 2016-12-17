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

class OptionalPrinter:
  """Print an llvm::Optional object."""

  def __init__(self, value):
    self.value = value

  class _iterator:
    def __init__(self, member, empty):
      self.member = member
      self.done = empty

    def __iter__(self):
      return self

    def next(self):
      if self.done:
        raise StopIteration
      self.done = True
      return ('value', self.member.dereference())

  def children(self):
    if not self.value['hasVal']:
      return self._iterator('', True)
    return self._iterator(self.value['storage']['buffer'].address.cast(self.value.type.template_argument(0).pointer()), False)

  def to_string(self):
    return 'llvm::Optional is %sinitialized' % ('' if self.value['hasVal'] else 'not ')

class DenseMapPrinter:
  "Print a DenseMap"

  class _iterator:
    def __init__(self, key_info_t, begin, end):
      self.key_info_t = key_info_t
      self.cur = begin
      self.end = end
      self.advancePastEmptyBuckets()
      self.first = True

    def __iter__(self):
      return self

    def advancePastEmptyBuckets(self):
      # disabled until the comments below can be addressed
      # keeping as notes/posterity/hints for future contributors
      return
      n = self.key_info_t.name
      is_equal = gdb.parse_and_eval(n + '::isEqual')
      empty = gdb.parse_and_eval(n + '::getEmptyKey()')
      tombstone = gdb.parse_and_eval(n + '::getTombstoneKey()')
      # the following is invalid, GDB fails with:
      #   Python Exception <class 'gdb.error'> Attempt to take address of value
      #   not located in memory.
      # because isEqual took parameter (for the unsigned long key I was testing)
      # by const ref, and GDB
      # It's also not entirely general - we should be accessing the "getFirst()"
      # member function, not the 'first' member variable, but I've yet to figure
      # out how to find/call member functions (especially (const) overloaded
      # ones) on a gdb.Value.
      while self.cur != self.end and (is_equal(self.cur.dereference()['first'], empty) or is_equal(self.cur.dereference()['first'], tombstone)):
        self.cur = self.cur + 1

    def next(self):
      if self.cur == self.end:
        raise StopIteration
      cur = self.cur
      v = cur.dereference()['first' if self.first else 'second']
      if not self.first:
        self.cur = self.cur + 1
        self.advancePastEmptyBuckets()
        self.first = True
      else:
        self.first = False
      return 'x', v

  def __init__(self, val):
    self.val = val

  def children(self):
    t = self.val.type.template_argument(3).pointer()
    begin = self.val['Buckets'].cast(t)
    end = (begin + self.val['NumBuckets']).cast(t)
    return self._iterator(self.val.type.template_argument(2), begin, end)

  def to_string(self):
    return 'llvm::DenseMap with %d elements' % (self.val['NumEntries'])

  def display_hint(self):
    return 'map'


pp = gdb.printing.RegexpCollectionPrettyPrinter("LLVMSupport")
pp.add_printer('llvm::SmallString', '^llvm::SmallString<.*>$', SmallStringPrinter)
pp.add_printer('llvm::StringRef', '^llvm::StringRef$', StringRefPrinter)
pp.add_printer('llvm::SmallVectorImpl', '^llvm::SmallVector(Impl)?<.*>$', SmallVectorPrinter)
pp.add_printer('llvm::ArrayRef', '^llvm::(Const)?ArrayRef<.*>$', ArrayRefPrinter)
pp.add_printer('llvm::Optional', '^llvm::Optional<.*>$', OptionalPrinter)
pp.add_printer('llvm::DenseMap', '^llvm::DenseMap<.*>$', DenseMapPrinter)
gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)
