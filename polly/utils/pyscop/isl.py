from ctypes import *

isl = cdll.LoadLibrary("libisl.so")


class Context:
    defaultInstance = None
    instances = {}

    def __init__(self):
        ptr = isl.isl_ctx_alloc()
        self.ptr = ptr
        Context.instances[ptr] = self

    def __del__(self):
        isl.isl_ctx_free(self)

    def from_param(self):
        return self.ptr

    @staticmethod
    def from_ptr(ptr):
        return Context.instances[ptr]

    @staticmethod
    def getDefaultInstance():
        if Context.defaultInstance == None:
            Context.defaultInstance = Context()

        return Context.defaultInstance


class IslObject:
    def __init__(self, string="", ctx=None, ptr=None):
        self.initialize_isl_methods()
        if ptr != None:
            self.ptr = ptr
            self.ctx = self.get_isl_method("get_ctx")(self)
            return

        if ctx == None:
            ctx = Context.getDefaultInstance()

        self.ctx = ctx
        self.ptr = self.get_isl_method("read_from_str")(ctx, string, -1)

    def __del__(self):
        self.get_isl_method("free")(self)

    def from_param(self):
        return self.ptr

    @property
    def context(self):
        return self.ctx

    def __repr__(self):
        p = Printer(self.ctx)
        self.to_printer(p)
        return p.getString()

    def __str__(self):
        p = Printer(self.ctx)
        self.to_printer(p)
        return p.getString()

    @staticmethod
    def isl_name():
        return "No isl name available"

    def initialize_isl_methods(self):
        if hasattr(self.__class__, "initialized"):
            return

        self.__class__.initalized = True
        self.get_isl_method("read_from_str").argtypes = [Context, c_char_p, c_int]
        self.get_isl_method("copy").argtypes = [self.__class__]
        self.get_isl_method("copy").restype = c_int
        self.get_isl_method("free").argtypes = [self.__class__]
        self.get_isl_method("get_ctx").argtypes = [self.__class__]
        self.get_isl_method("get_ctx").restype = Context.from_ptr
        getattr(isl, "isl_printer_print_" + self.isl_name()).argtypes = [
            Printer,
            self.__class__,
        ]

    def get_isl_method(self, name):
        return getattr(isl, "isl_" + self.isl_name() + "_" + name)

    def to_printer(self, printer):
        getattr(isl, "isl_printer_print_" + self.isl_name())(printer, self)


class BSet(IslObject):
    @staticmethod
    def from_ptr(ptr):
        if not ptr:
            return
        return BSet(ptr=ptr)

    @staticmethod
    def isl_name():
        return "basic_set"


class Set(IslObject):
    @staticmethod
    def from_ptr(ptr):
        if not ptr:
            return
        return Set(ptr=ptr)

    @staticmethod
    def isl_name():
        return "set"


class USet(IslObject):
    @staticmethod
    def from_ptr(ptr):
        if not ptr:
            return
        return USet(ptr=ptr)

    @staticmethod
    def isl_name():
        return "union_set"


class BMap(IslObject):
    @staticmethod
    def from_ptr(ptr):
        if not ptr:
            return
        return BMap(ptr=ptr)

    def __mul__(self, set):
        return self.intersect_domain(set)

    @staticmethod
    def isl_name():
        return "basic_map"


class Map(IslObject):
    @staticmethod
    def from_ptr(ptr):
        if not ptr:
            return
        return Map(ptr=ptr)

    def __mul__(self, set):
        return self.intersect_domain(set)

    @staticmethod
    def isl_name():
        return "map"

    @staticmethod
    def lex_lt(dim):
        dim = isl.isl_dim_copy(dim)
        return isl.isl_map_lex_lt(dim)

    @staticmethod
    def lex_le(dim):
        dim = isl.isl_dim_copy(dim)
        return isl.isl_map_lex_le(dim)

    @staticmethod
    def lex_gt(dim):
        dim = isl.isl_dim_copy(dim)
        return isl.isl_map_lex_gt(dim)

    @staticmethod
    def lex_ge(dim):
        dim = isl.isl_dim_copy(dim)
        return isl.isl_map_lex_ge(dim)


class UMap(IslObject):
    @staticmethod
    def from_ptr(ptr):
        if not ptr:
            return
        return UMap(ptr=ptr)

    @staticmethod
    def isl_name():
        return "union_map"


class Dim(IslObject):
    @staticmethod
    def from_ptr(ptr):
        if not ptr:
            return
        return Dim(ptr=ptr)

    @staticmethod
    def isl_name():
        return "dim"

    def initialize_isl_methods(self):
        if hasattr(self.__class__, "initialized"):
            return

        self.__class__.initalized = True
        self.get_isl_method("copy").argtypes = [self.__class__]
        self.get_isl_method("copy").restype = c_int
        self.get_isl_method("free").argtypes = [self.__class__]
        self.get_isl_method("get_ctx").argtypes = [self.__class__]
        self.get_isl_method("get_ctx").restype = Context.from_ptr

    def __repr__(self):
        return str(self)

    def __str__(self):

        dimParam = isl.isl_dim_size(self, 1)
        dimIn = isl.isl_dim_size(self, 2)
        dimOut = isl.isl_dim_size(self, 3)

        if dimIn:
            return "<dim In:%s, Out:%s, Param:%s>" % (dimIn, dimOut, dimParam)

        return "<dim Set:%s, Param:%s>" % (dimOut, dimParam)


class Printer:
    FORMAT_ISL = 0
    FORMAT_POLYLIB = 1
    FORMAT_POLYLIB_CONSTRAINTS = 2
    FORMAT_OMEGA = 3
    FORMAT_C = 4
    FORMAT_LATEX = 5
    FORMAT_EXT_POLYLIB = 6

    def __init__(self, ctx=None):
        if ctx == None:
            ctx = Context.getDefaultInstance()

        self.ctx = ctx
        self.ptr = isl.isl_printer_to_str(ctx)

    def setFormat(self, format):
        self.ptr = isl.isl_printer_set_output_format(self, format)

    def from_param(self):
        return self.ptr

    def __del__(self):
        isl.isl_printer_free(self)

    def getString(self):
        return isl.isl_printer_get_str(self)


functions = [
    # Unary properties
    ("is_empty", BSet, [BSet], c_int),
    ("is_empty", Set, [Set], c_int),
    ("is_empty", USet, [USet], c_int),
    ("is_empty", BMap, [BMap], c_int),
    ("is_empty", Map, [Map], c_int),
    ("is_empty", UMap, [UMap], c_int),
    #         ("is_universe", Set, [Set], c_int),
    #         ("is_universe", Map, [Map], c_int),
    ("is_single_valued", Map, [Map], c_int),
    ("is_bijective", Map, [Map], c_int),
    ("is_wrapping", BSet, [BSet], c_int),
    ("is_wrapping", Set, [Set], c_int),
    # Binary properties
    ("is_equal", BSet, [BSet, BSet], c_int),
    ("is_equal", Set, [Set, Set], c_int),
    ("is_equal", USet, [USet, USet], c_int),
    ("is_equal", BMap, [BMap, BMap], c_int),
    ("is_equal", Map, [Map, Map], c_int),
    ("is_equal", UMap, [UMap, UMap], c_int),
    # is_disjoint missing
    # ("is_subset", BSet, [BSet, BSet], c_int),
    ("is_subset", Set, [Set, Set], c_int),
    ("is_subset", USet, [USet, USet], c_int),
    ("is_subset", BMap, [BMap, BMap], c_int),
    ("is_subset", Map, [Map, Map], c_int),
    ("is_subset", UMap, [UMap, UMap], c_int),
    # ("is_strict_subset", BSet, [BSet, BSet], c_int),
    ("is_strict_subset", Set, [Set, Set], c_int),
    ("is_strict_subset", USet, [USet, USet], c_int),
    ("is_strict_subset", BMap, [BMap, BMap], c_int),
    ("is_strict_subset", Map, [Map, Map], c_int),
    ("is_strict_subset", UMap, [UMap, UMap], c_int),
    # Unary Operations
    ("complement", Set, [Set], Set),
    ("reverse", BMap, [BMap], BMap),
    ("reverse", Map, [Map], Map),
    ("reverse", UMap, [UMap], UMap),
    # Projection missing
    ("range", BMap, [BMap], BSet),
    ("range", Map, [Map], Set),
    ("range", UMap, [UMap], USet),
    ("domain", BMap, [BMap], BSet),
    ("domain", Map, [Map], Set),
    ("domain", UMap, [UMap], USet),
    ("identity", Set, [Set], Map),
    ("identity", USet, [USet], UMap),
    ("deltas", BMap, [BMap], BSet),
    ("deltas", Map, [Map], Set),
    ("deltas", UMap, [UMap], USet),
    ("coalesce", Set, [Set], Set),
    ("coalesce", USet, [USet], USet),
    ("coalesce", Map, [Map], Map),
    ("coalesce", UMap, [UMap], UMap),
    ("detect_equalities", BSet, [BSet], BSet),
    ("detect_equalities", Set, [Set], Set),
    ("detect_equalities", USet, [USet], USet),
    ("detect_equalities", BMap, [BMap], BMap),
    ("detect_equalities", Map, [Map], Map),
    ("detect_equalities", UMap, [UMap], UMap),
    ("convex_hull", Set, [Set], Set),
    ("convex_hull", Map, [Map], Map),
    ("simple_hull", Set, [Set], Set),
    ("simple_hull", Map, [Map], Map),
    ("affine_hull", BSet, [BSet], BSet),
    ("affine_hull", Set, [Set], BSet),
    ("affine_hull", USet, [USet], USet),
    ("affine_hull", BMap, [BMap], BMap),
    ("affine_hull", Map, [Map], BMap),
    ("affine_hull", UMap, [UMap], UMap),
    ("polyhedral_hull", Set, [Set], Set),
    ("polyhedral_hull", USet, [USet], USet),
    ("polyhedral_hull", Map, [Map], Map),
    ("polyhedral_hull", UMap, [UMap], UMap),
    # Power missing
    # Transitive closure missing
    # Reaching path lengths missing
    ("wrap", BMap, [BMap], BSet),
    ("wrap", Map, [Map], Set),
    ("wrap", UMap, [UMap], USet),
    ("unwrap", BSet, [BMap], BMap),
    ("unwrap", Set, [Map], Map),
    ("unwrap", USet, [UMap], UMap),
    ("flatten", Set, [Set], Set),
    ("flatten", Map, [Map], Map),
    ("flatten_map", Set, [Set], Map),
    # Dimension manipulation missing
    # Binary Operations
    ("intersect", BSet, [BSet, BSet], BSet),
    ("intersect", Set, [Set, Set], Set),
    ("intersect", USet, [USet, USet], USet),
    ("intersect", BMap, [BMap, BMap], BMap),
    ("intersect", Map, [Map, Map], Map),
    ("intersect", UMap, [UMap, UMap], UMap),
    ("intersect_domain", BMap, [BMap, BSet], BMap),
    ("intersect_domain", Map, [Map, Set], Map),
    ("intersect_domain", UMap, [UMap, USet], UMap),
    ("intersect_range", BMap, [BMap, BSet], BMap),
    ("intersect_range", Map, [Map, Set], Map),
    ("intersect_range", UMap, [UMap, USet], UMap),
    ("union", BSet, [BSet, BSet], Set),
    ("union", Set, [Set, Set], Set),
    ("union", USet, [USet, USet], USet),
    ("union", BMap, [BMap, BMap], Map),
    ("union", Map, [Map, Map], Map),
    ("union", UMap, [UMap, UMap], UMap),
    ("subtract", Set, [Set, Set], Set),
    ("subtract", Map, [Map, Map], Map),
    ("subtract", USet, [USet, USet], USet),
    ("subtract", UMap, [UMap, UMap], UMap),
    ("apply", BSet, [BSet, BMap], BSet),
    ("apply", Set, [Set, Map], Set),
    ("apply", USet, [USet, UMap], USet),
    ("apply_domain", BMap, [BMap, BMap], BMap),
    ("apply_domain", Map, [Map, Map], Map),
    ("apply_domain", UMap, [UMap, UMap], UMap),
    ("apply_range", BMap, [BMap, BMap], BMap),
    ("apply_range", Map, [Map, Map], Map),
    ("apply_range", UMap, [UMap, UMap], UMap),
    ("gist", BSet, [BSet, BSet], BSet),
    ("gist", Set, [Set, Set], Set),
    ("gist", USet, [USet, USet], USet),
    ("gist", BMap, [BMap, BMap], BMap),
    ("gist", Map, [Map, Map], Map),
    ("gist", UMap, [UMap, UMap], UMap),
    # Lexicographic Optimizations
    # partial_lexmin missing
    ("lexmin", BSet, [BSet], BSet),
    ("lexmin", Set, [Set], Set),
    ("lexmin", USet, [USet], USet),
    ("lexmin", BMap, [BMap], BMap),
    ("lexmin", Map, [Map], Map),
    ("lexmin", UMap, [UMap], UMap),
    ("lexmax", BSet, [BSet], BSet),
    ("lexmax", Set, [Set], Set),
    ("lexmax", USet, [USet], USet),
    ("lexmax", BMap, [BMap], BMap),
    ("lexmax", Map, [Map], Map),
    ("lexmax", UMap, [UMap], UMap),
    # Undocumented
    ("lex_lt_union_set", USet, [USet, USet], UMap),
    ("lex_le_union_set", USet, [USet, USet], UMap),
    ("lex_gt_union_set", USet, [USet, USet], UMap),
    ("lex_ge_union_set", USet, [USet, USet], UMap),
]
keep_functions = [
    # Unary properties
    ("get_dim", BSet, [BSet], Dim),
    ("get_dim", Set, [Set], Dim),
    ("get_dim", USet, [USet], Dim),
    ("get_dim", BMap, [BMap], Dim),
    ("get_dim", Map, [Map], Dim),
    ("get_dim", UMap, [UMap], Dim),
]


def addIslFunction(object, name):
    functionName = "isl_" + object.isl_name() + "_" + name
    islFunction = getattr(isl, functionName)
    if len(islFunction.argtypes) == 1:
        f = lambda a: islFunctionOneOp(islFunction, a)
    elif len(islFunction.argtypes) == 2:
        f = lambda a, b: islFunctionTwoOp(islFunction, a, b)
    object.__dict__[name] = f


def islFunctionOneOp(islFunction, ops):
    ops = getattr(isl, "isl_" + ops.isl_name() + "_copy")(ops)
    return islFunction(ops)


def islFunctionTwoOp(islFunction, opOne, opTwo):
    opOne = getattr(isl, "isl_" + opOne.isl_name() + "_copy")(opOne)
    opTwo = getattr(isl, "isl_" + opTwo.isl_name() + "_copy")(opTwo)
    return islFunction(opOne, opTwo)


for (operation, base, operands, ret) in functions:
    functionName = "isl_" + base.isl_name() + "_" + operation
    islFunction = getattr(isl, functionName)
    if len(operands) == 1:
        islFunction.argtypes = [c_int]
    elif len(operands) == 2:
        islFunction.argtypes = [c_int, c_int]

    if ret == c_int:
        islFunction.restype = ret
    else:
        islFunction.restype = ret.from_ptr

    addIslFunction(base, operation)


def addIslFunctionKeep(object, name):
    functionName = "isl_" + object.isl_name() + "_" + name
    islFunction = getattr(isl, functionName)
    if len(islFunction.argtypes) == 1:
        f = lambda a: islFunctionOneOpKeep(islFunction, a)
    elif len(islFunction.argtypes) == 2:
        f = lambda a, b: islFunctionTwoOpKeep(islFunction, a, b)
    object.__dict__[name] = f


def islFunctionOneOpKeep(islFunction, ops):
    return islFunction(ops)


def islFunctionTwoOpKeep(islFunction, opOne, opTwo):
    return islFunction(opOne, opTwo)


for (operation, base, operands, ret) in keep_functions:
    functionName = "isl_" + base.isl_name() + "_" + operation
    islFunction = getattr(isl, functionName)
    if len(operands) == 1:
        islFunction.argtypes = [c_int]
    elif len(operands) == 2:
        islFunction.argtypes = [c_int, c_int]

    if ret == c_int:
        islFunction.restype = ret
    else:
        islFunction.restype = ret.from_ptr

    addIslFunctionKeep(base, operation)

isl.isl_ctx_free.argtypes = [Context]
isl.isl_basic_set_read_from_str.argtypes = [Context, c_char_p, c_int]
isl.isl_set_read_from_str.argtypes = [Context, c_char_p, c_int]
isl.isl_basic_set_copy.argtypes = [BSet]
isl.isl_basic_set_copy.restype = c_int
isl.isl_set_copy.argtypes = [Set]
isl.isl_set_copy.restype = c_int
isl.isl_set_copy.argtypes = [Set]
isl.isl_set_copy.restype = c_int
isl.isl_set_free.argtypes = [Set]
isl.isl_basic_set_get_ctx.argtypes = [BSet]
isl.isl_basic_set_get_ctx.restype = Context.from_ptr
isl.isl_set_get_ctx.argtypes = [Set]
isl.isl_set_get_ctx.restype = Context.from_ptr
isl.isl_basic_set_get_dim.argtypes = [BSet]
isl.isl_basic_set_get_dim.restype = Dim.from_ptr
isl.isl_set_get_dim.argtypes = [Set]
isl.isl_set_get_dim.restype = Dim.from_ptr
isl.isl_union_set_get_dim.argtypes = [USet]
isl.isl_union_set_get_dim.restype = Dim.from_ptr

isl.isl_basic_map_read_from_str.argtypes = [Context, c_char_p, c_int]
isl.isl_map_read_from_str.argtypes = [Context, c_char_p, c_int]
isl.isl_basic_map_free.argtypes = [BMap]
isl.isl_map_free.argtypes = [Map]
isl.isl_basic_map_copy.argtypes = [BMap]
isl.isl_basic_map_copy.restype = c_int
isl.isl_map_copy.argtypes = [Map]
isl.isl_map_copy.restype = c_int
isl.isl_map_get_ctx.argtypes = [Map]
isl.isl_basic_map_get_ctx.argtypes = [BMap]
isl.isl_basic_map_get_ctx.restype = Context.from_ptr
isl.isl_map_get_ctx.argtypes = [Map]
isl.isl_map_get_ctx.restype = Context.from_ptr
isl.isl_basic_map_get_dim.argtypes = [BMap]
isl.isl_basic_map_get_dim.restype = Dim.from_ptr
isl.isl_map_get_dim.argtypes = [Map]
isl.isl_map_get_dim.restype = Dim.from_ptr
isl.isl_union_map_get_dim.argtypes = [UMap]
isl.isl_union_map_get_dim.restype = Dim.from_ptr
isl.isl_printer_free.argtypes = [Printer]
isl.isl_printer_to_str.argtypes = [Context]
isl.isl_printer_print_basic_set.argtypes = [Printer, BSet]
isl.isl_printer_print_set.argtypes = [Printer, Set]
isl.isl_printer_print_basic_map.argtypes = [Printer, BMap]
isl.isl_printer_print_map.argtypes = [Printer, Map]
isl.isl_printer_get_str.argtypes = [Printer]
isl.isl_printer_get_str.restype = c_char_p
isl.isl_printer_set_output_format.argtypes = [Printer, c_int]
isl.isl_printer_set_output_format.restype = c_int
isl.isl_dim_size.argtypes = [Dim, c_int]
isl.isl_dim_size.restype = c_int

isl.isl_map_lex_lt.argtypes = [c_int]
isl.isl_map_lex_lt.restype = Map.from_ptr
isl.isl_map_lex_le.argtypes = [c_int]
isl.isl_map_lex_le.restype = Map.from_ptr
isl.isl_map_lex_gt.argtypes = [c_int]
isl.isl_map_lex_gt.restype = Map.from_ptr
isl.isl_map_lex_ge.argtypes = [c_int]
isl.isl_map_lex_ge.restype = Map.from_ptr

isl.isl_union_map_compute_flow.argtypes = [
    c_int,
    c_int,
    c_int,
    c_int,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]


def dependences(sink, must_source, may_source, schedule):
    sink = getattr(isl, "isl_" + sink.isl_name() + "_copy")(sink)
    must_source = getattr(isl, "isl_" + must_source.isl_name() + "_copy")(must_source)
    may_source = getattr(isl, "isl_" + may_source.isl_name() + "_copy")(may_source)
    schedule = getattr(isl, "isl_" + schedule.isl_name() + "_copy")(schedule)
    must_dep = c_int()
    may_dep = c_int()
    must_no_source = c_int()
    may_no_source = c_int()
    isl.isl_union_map_compute_flow(
        sink,
        must_source,
        may_source,
        schedule,
        byref(must_dep),
        byref(may_dep),
        byref(must_no_source),
        byref(may_no_source),
    )

    return (
        UMap.from_ptr(must_dep),
        UMap.from_ptr(may_dep),
        USet.from_ptr(must_no_source),
        USet.from_ptr(may_no_source),
    )


__all__ = ["Set", "Map", "Printer", "Context"]
