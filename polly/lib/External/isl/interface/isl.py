isl_dlname='libisl.so.23'
import os
from ctypes import *
from ctypes.util import find_library

isl_dyld_library_path = os.environ.get('ISL_DYLD_LIBRARY_PATH')
if isl_dyld_library_path != None:
    os.environ['DYLD_LIBRARY_PATH'] =  isl_dyld_library_path
try:
    isl = cdll.LoadLibrary(isl_dlname)
except:
    isl = cdll.LoadLibrary(find_library("isl"))
libc = cdll.LoadLibrary(find_library("c"))

class Error(Exception):
    pass

class Context:
    defaultInstance = None

    def __init__(self):
        ptr = isl.isl_ctx_alloc()
        self.ptr = ptr

    def __del__(self):
        isl.isl_ctx_free(self)

    def from_param(self):
        return c_void_p(self.ptr)

    @staticmethod
    def getDefaultInstance():
        if Context.defaultInstance == None:
            Context.defaultInstance = Context()
        return Context.defaultInstance

isl.isl_ctx_alloc.restype = c_void_p
isl.isl_ctx_free.argtypes = [Context]

class union_pw_multi_aff(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_multi_aff_from_multi_aff(isl.isl_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_multi_aff_from_pw_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is union_pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_multi_aff_from_union_pw_aff(isl.isl_union_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_multi_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_pw_multi_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ptr = isl.isl_union_pw_multi_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_pw_multi_aff("""%s""")' % s
        else:
            return 'isl.union_pw_multi_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_add(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def apply(*args):
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_pw_multi_aff_apply_union_pw_multi_aff(isl.isl_union_pw_multi_aff_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def as_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_as_pw_multi_aff(isl.isl_union_pw_multi_aff_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_coalesce(isl.isl_union_pw_multi_aff_copy(arg0.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_domain(isl.isl_union_pw_multi_aff_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def empty(*args):
        if len(args) == 0:
            ctx = Context.getDefaultInstance()
            res = isl.isl_union_pw_multi_aff_empty_ctx(ctx)
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def extract_pw_multi_aff(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_extract_pw_multi_aff(arg0.ptr, isl.isl_space_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_flat_range_product(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def space(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_gist(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(*args):
        if len(args) == 2 and args[1].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_union_pw_multi_aff_intersect_domain_space(isl.isl_union_pw_multi_aff_copy(args[0].ptr), isl.isl_space_copy(args[1].ptr))
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_set:
            ctx = args[0].ctx
            res = isl.isl_union_pw_multi_aff_intersect_domain_union_set(isl.isl_union_pw_multi_aff_copy(args[0].ptr), isl.isl_union_set_copy(args[1].ptr))
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def intersect_domain_wrapped_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_intersect_domain_wrapped_domain(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain_wrapped_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_intersect_domain_wrapped_range(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_intersect_params(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def involves_locals(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_involves_locals(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def isa_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_isa_pw_multi_aff(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def plain_is_empty(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_plain_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def preimage_domain_wrapped_domain(*args):
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff(isl.isl_union_pw_multi_aff_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_pw_multi_aff_pullback_union_pw_multi_aff(isl.isl_union_pw_multi_aff_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def range_factor_domain(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_range_factor_domain(isl.isl_union_pw_multi_aff_copy(arg0.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def range_factor_range(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_range_factor_range(isl.isl_union_pw_multi_aff_copy(arg0.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_range_product(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_sub(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def subtract_domain(*args):
        if len(args) == 2 and args[1].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_union_pw_multi_aff_subtract_domain_space(isl.isl_union_pw_multi_aff_copy(args[0].ptr), isl.isl_space_copy(args[1].ptr))
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_set:
            ctx = args[0].ctx
            res = isl.isl_union_pw_multi_aff_subtract_domain_union_set(isl.isl_union_pw_multi_aff_copy(args[0].ptr), isl.isl_union_set_copy(args[1].ptr))
            obj = union_pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_union_add(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj

isl.isl_union_pw_multi_aff_from_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_from_multi_aff.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_from_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_from_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_from_union_pw_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_from_union_pw_aff.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_read_from_str.restype = c_void_p
isl.isl_union_pw_multi_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_pw_multi_aff_add.restype = c_void_p
isl.isl_union_pw_multi_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_apply_union_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_apply_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_as_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_as_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_coalesce.restype = c_void_p
isl.isl_union_pw_multi_aff_coalesce.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_domain.restype = c_void_p
isl.isl_union_pw_multi_aff_domain.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_empty_ctx.restype = c_void_p
isl.isl_union_pw_multi_aff_empty_ctx.argtypes = [Context]
isl.isl_union_pw_multi_aff_extract_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_extract_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_flat_range_product.restype = c_void_p
isl.isl_union_pw_multi_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_get_space.restype = c_void_p
isl.isl_union_pw_multi_aff_get_space.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_gist.restype = c_void_p
isl.isl_union_pw_multi_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_intersect_domain_space.restype = c_void_p
isl.isl_union_pw_multi_aff_intersect_domain_space.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_intersect_domain_union_set.restype = c_void_p
isl.isl_union_pw_multi_aff_intersect_domain_union_set.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_intersect_domain_wrapped_domain.restype = c_void_p
isl.isl_union_pw_multi_aff_intersect_domain_wrapped_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_intersect_domain_wrapped_range.restype = c_void_p
isl.isl_union_pw_multi_aff_intersect_domain_wrapped_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_intersect_params.restype = c_void_p
isl.isl_union_pw_multi_aff_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_involves_locals.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_isa_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_plain_is_empty.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_range_factor_domain.restype = c_void_p
isl.isl_union_pw_multi_aff_range_factor_domain.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_range_factor_range.restype = c_void_p
isl.isl_union_pw_multi_aff_range_factor_range.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_range_product.restype = c_void_p
isl.isl_union_pw_multi_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_sub.restype = c_void_p
isl.isl_union_pw_multi_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_subtract_domain_space.restype = c_void_p
isl.isl_union_pw_multi_aff_subtract_domain_space.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_subtract_domain_union_set.restype = c_void_p
isl.isl_union_pw_multi_aff_subtract_domain_union_set.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_union_add.restype = c_void_p
isl.isl_union_pw_multi_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_copy.restype = c_void_p
isl.isl_union_pw_multi_aff_copy.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_free.restype = c_void_p
isl.isl_union_pw_multi_aff_free.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_to_str.restype = POINTER(c_char)
isl.isl_union_pw_multi_aff_to_str.argtypes = [c_void_p]

class multi_union_pw_aff(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is multi_pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_union_pw_aff_from_multi_pw_aff(isl.isl_multi_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is union_pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_union_pw_aff_from_union_pw_aff(isl.isl_union_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 2 and args[0].__class__ is space and args[1].__class__ is union_pw_aff_list:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_union_pw_aff_from_union_pw_aff_list(isl.isl_space_copy(args[0].ptr), isl.isl_union_pw_aff_list_copy(args[1].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_union_pw_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_union_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_multi_union_pw_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_union_pw_aff("""%s""")' % s
        else:
            return 'isl.multi_union_pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_add(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def bind(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_bind(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_coalesce(isl.isl_multi_union_pw_aff_copy(arg0.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_domain(isl.isl_multi_union_pw_aff_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_flat_range_product(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_get_at(arg0.ptr, arg1)
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def list(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_get_list(arg0.ptr)
        obj = union_pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def get_list(arg0):
        return arg0.list()
    def space(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_gist(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_intersect_domain(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_intersect_params(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def involves_nan(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_involves_nan(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def neg(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_neg(isl.isl_multi_union_pw_aff_copy(arg0.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def plain_is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_plain_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_multi_union_pw_aff_pullback_union_pw_multi_aff(isl.isl_multi_union_pw_aff_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = multi_union_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_range_product(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def scale(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_union_pw_aff_scale_multi_val(isl.isl_multi_union_pw_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_union_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_union_pw_aff_scale_val(isl.isl_multi_union_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_union_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale_down(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_union_pw_aff_scale_down_multi_val(isl.isl_multi_union_pw_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_union_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_union_pw_aff_scale_down_val(isl.isl_multi_union_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_union_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def set_at(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is union_pw_aff:
                arg2 = union_pw_aff(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_set_at(isl.isl_multi_union_pw_aff_copy(arg0.ptr), arg1, isl.isl_union_pw_aff_copy(arg2.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_sub(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_union_add(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def zero(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_zero(isl.isl_space_copy(arg0.ptr))
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj

isl.isl_multi_union_pw_aff_from_multi_pw_aff.restype = c_void_p
isl.isl_multi_union_pw_aff_from_multi_pw_aff.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_from_union_pw_aff.restype = c_void_p
isl.isl_multi_union_pw_aff_from_union_pw_aff.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_from_union_pw_aff_list.restype = c_void_p
isl.isl_multi_union_pw_aff_from_union_pw_aff_list.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_read_from_str.restype = c_void_p
isl.isl_multi_union_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_union_pw_aff_add.restype = c_void_p
isl.isl_multi_union_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_bind.restype = c_void_p
isl.isl_multi_union_pw_aff_bind.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_coalesce.restype = c_void_p
isl.isl_multi_union_pw_aff_coalesce.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_domain.restype = c_void_p
isl.isl_multi_union_pw_aff_domain.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_flat_range_product.restype = c_void_p
isl.isl_multi_union_pw_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_get_at.restype = c_void_p
isl.isl_multi_union_pw_aff_get_at.argtypes = [c_void_p, c_int]
isl.isl_multi_union_pw_aff_get_list.restype = c_void_p
isl.isl_multi_union_pw_aff_get_list.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_get_space.restype = c_void_p
isl.isl_multi_union_pw_aff_get_space.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_gist.restype = c_void_p
isl.isl_multi_union_pw_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_intersect_domain.restype = c_void_p
isl.isl_multi_union_pw_aff_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_intersect_params.restype = c_void_p
isl.isl_multi_union_pw_aff_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_involves_nan.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_neg.restype = c_void_p
isl.isl_multi_union_pw_aff_neg.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_plain_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_multi_union_pw_aff_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_range_product.restype = c_void_p
isl.isl_multi_union_pw_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_scale_multi_val.restype = c_void_p
isl.isl_multi_union_pw_aff_scale_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_scale_val.restype = c_void_p
isl.isl_multi_union_pw_aff_scale_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_scale_down_multi_val.restype = c_void_p
isl.isl_multi_union_pw_aff_scale_down_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_scale_down_val.restype = c_void_p
isl.isl_multi_union_pw_aff_scale_down_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_set_at.restype = c_void_p
isl.isl_multi_union_pw_aff_set_at.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_multi_union_pw_aff_size.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_sub.restype = c_void_p
isl.isl_multi_union_pw_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_union_add.restype = c_void_p
isl.isl_multi_union_pw_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_zero.restype = c_void_p
isl.isl_multi_union_pw_aff_zero.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_copy.restype = c_void_p
isl.isl_multi_union_pw_aff_copy.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_free.restype = c_void_p
isl.isl_multi_union_pw_aff_free.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_multi_union_pw_aff_to_str.argtypes = [c_void_p]

class union_pw_aff(union_pw_multi_aff, multi_union_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_aff_from_aff(isl.isl_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_aff_from_pw_aff(isl.isl_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_union_pw_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_pw_aff("""%s""")' % s
        else:
            return 'isl.union_pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_aff:
                arg1 = union_pw_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_add(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_pw_aff_copy(arg1.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def bind(*args):
        if len(args) == 2 and (args[1].__class__ is id or type(args[1]) == str):
            args = list(args)
            try:
                if not args[1].__class__ is id:
                    args[1] = id(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_union_pw_aff_bind_id(isl.isl_union_pw_aff_copy(args[0].ptr), isl.isl_id_copy(args[1].ptr))
            obj = union_set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def coalesce(arg0):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_coalesce(isl.isl_union_pw_aff_copy(arg0.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_domain(isl.isl_union_pw_aff_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def space(arg0):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            return union_pw_multi_aff(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_gist(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(*args):
        if len(args) == 2 and args[1].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_union_pw_aff_intersect_domain_space(isl.isl_union_pw_aff_copy(args[0].ptr), isl.isl_space_copy(args[1].ptr))
            obj = union_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_set:
            ctx = args[0].ctx
            res = isl.isl_union_pw_aff_intersect_domain_union_set(isl.isl_union_pw_aff_copy(args[0].ptr), isl.isl_union_set_copy(args[1].ptr))
            obj = union_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def intersect_domain_wrapped_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            return union_pw_multi_aff(arg0).intersect_domain_wrapped_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_intersect_domain_wrapped_domain(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain_wrapped_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            return union_pw_multi_aff(arg0).intersect_domain_wrapped_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_intersect_domain_wrapped_range(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_multi_aff(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_intersect_params(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_pw_aff_pullback_union_pw_multi_aff(isl.isl_union_pw_aff_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = union_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_aff:
                arg1 = union_pw_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).sub(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_sub(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_pw_aff_copy(arg1.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def subtract_domain(*args):
        if len(args) == 2 and args[1].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_union_pw_aff_subtract_domain_space(isl.isl_union_pw_aff_copy(args[0].ptr), isl.isl_space_copy(args[1].ptr))
            obj = union_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_set:
            ctx = args[0].ctx
            res = isl.isl_union_pw_aff_subtract_domain_union_set(isl.isl_union_pw_aff_copy(args[0].ptr), isl.isl_union_set_copy(args[1].ptr))
            obj = union_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_aff:
                arg1 = union_pw_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).union_add(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_union_add(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_pw_aff_copy(arg1.ptr))
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj

isl.isl_union_pw_aff_from_aff.restype = c_void_p
isl.isl_union_pw_aff_from_aff.argtypes = [c_void_p]
isl.isl_union_pw_aff_from_pw_aff.restype = c_void_p
isl.isl_union_pw_aff_from_pw_aff.argtypes = [c_void_p]
isl.isl_union_pw_aff_read_from_str.restype = c_void_p
isl.isl_union_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_pw_aff_add.restype = c_void_p
isl.isl_union_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_bind_id.restype = c_void_p
isl.isl_union_pw_aff_bind_id.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_coalesce.restype = c_void_p
isl.isl_union_pw_aff_coalesce.argtypes = [c_void_p]
isl.isl_union_pw_aff_domain.restype = c_void_p
isl.isl_union_pw_aff_domain.argtypes = [c_void_p]
isl.isl_union_pw_aff_get_space.restype = c_void_p
isl.isl_union_pw_aff_get_space.argtypes = [c_void_p]
isl.isl_union_pw_aff_gist.restype = c_void_p
isl.isl_union_pw_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_intersect_domain_space.restype = c_void_p
isl.isl_union_pw_aff_intersect_domain_space.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_intersect_domain_union_set.restype = c_void_p
isl.isl_union_pw_aff_intersect_domain_union_set.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_intersect_domain_wrapped_domain.restype = c_void_p
isl.isl_union_pw_aff_intersect_domain_wrapped_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_intersect_domain_wrapped_range.restype = c_void_p
isl.isl_union_pw_aff_intersect_domain_wrapped_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_intersect_params.restype = c_void_p
isl.isl_union_pw_aff_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_aff_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_sub.restype = c_void_p
isl.isl_union_pw_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_subtract_domain_space.restype = c_void_p
isl.isl_union_pw_aff_subtract_domain_space.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_subtract_domain_union_set.restype = c_void_p
isl.isl_union_pw_aff_subtract_domain_union_set.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_union_add.restype = c_void_p
isl.isl_union_pw_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_copy.restype = c_void_p
isl.isl_union_pw_aff_copy.argtypes = [c_void_p]
isl.isl_union_pw_aff_free.restype = c_void_p
isl.isl_union_pw_aff_free.argtypes = [c_void_p]
isl.isl_union_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_union_pw_aff_to_str.argtypes = [c_void_p]

class multi_pw_aff(multi_union_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_aff(isl.isl_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_multi_aff(isl.isl_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_pw_aff(isl.isl_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 2 and args[0].__class__ is space and args[1].__class__ is pw_aff_list:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_pw_aff_list(isl.isl_space_copy(args[0].ptr), isl.isl_pw_aff_list_copy(args[1].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_pw_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_multi_pw_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_pw_aff("""%s""")' % s
        else:
            return 'isl.multi_pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_add(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def add_constant(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_add_constant_multi_val(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_add_constant_val(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def bind(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return multi_union_pw_aff(arg0).bind(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_bind(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def bind_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return multi_union_pw_aff(arg0).bind_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_bind_domain(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def bind_domain_wrapped_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return multi_union_pw_aff(arg0).bind_domain_wrapped_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_bind_domain_wrapped_domain(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_coalesce(isl.isl_multi_pw_aff_copy(arg0.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_domain(isl.isl_multi_pw_aff_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).flat_range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_flat_range_product(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_get_at(arg0.ptr, arg1)
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def list(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_get_list(arg0.ptr)
        obj = pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def get_list(arg0):
        return arg0.list()
    def space(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return multi_union_pw_aff(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_gist(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def identity(*args):
        if len(args) == 1:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_identity_multi_pw_aff(isl.isl_multi_pw_aff_copy(args[0].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    @staticmethod
    def identity_on_domain(*args):
        if len(args) == 1 and args[0].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_identity_on_domain_space(isl.isl_space_copy(args[0].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            return multi_union_pw_aff(arg0).insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_insert_domain(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_space_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return multi_union_pw_aff(arg0).intersect_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_intersect_domain(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return multi_union_pw_aff(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_intersect_params(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def involves_nan(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_involves_nan(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def involves_param(*args):
        if len(args) == 2 and (args[1].__class__ is id or type(args[1]) == str):
            args = list(args)
            try:
                if not args[1].__class__ is id:
                    args[1] = id(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_involves_param_id(args[0].ptr, args[1].ptr)
            if res < 0:
                raise
            return bool(res)
        if len(args) == 2 and args[1].__class__ is id_list:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_involves_param_id_list(args[0].ptr, args[1].ptr)
            if res < 0:
                raise
            return bool(res)
        raise Error
    def max(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).max(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_max(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def max_multi_val(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_max_multi_val(isl.isl_multi_pw_aff_copy(arg0.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def min(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).min(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_min(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def min_multi_val(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_min_multi_val(isl.isl_multi_pw_aff_copy(arg0.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def neg(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_neg(isl.isl_multi_pw_aff_copy(arg0.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def plain_is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).plain_is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_plain_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_product(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_pullback_multi_aff(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_pullback_multi_pw_aff(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_pullback_pw_multi_aff(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_range_product(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def scale(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_scale_multi_val(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_scale_val(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale_down(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_scale_down_multi_val(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_pw_aff_scale_down_val(isl.isl_multi_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def set_at(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is pw_aff:
                arg2 = pw_aff(arg2)
        except:
            return multi_union_pw_aff(arg0).set_at(arg1, arg2)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_set_at(isl.isl_multi_pw_aff_copy(arg0.ptr), arg1, isl.isl_pw_aff_copy(arg2.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).sub(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_sub(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def unbind_params_insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return multi_union_pw_aff(arg0).unbind_params_insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_unbind_params_insert_domain(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).union_add(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_union_add(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def zero(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_zero(isl.isl_space_copy(arg0.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj

isl.isl_multi_pw_aff_from_aff.restype = c_void_p
isl.isl_multi_pw_aff_from_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_from_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_from_multi_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_from_pw_aff.restype = c_void_p
isl.isl_multi_pw_aff_from_pw_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_from_pw_aff_list.restype = c_void_p
isl.isl_multi_pw_aff_from_pw_aff_list.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_from_pw_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_from_pw_multi_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_read_from_str.restype = c_void_p
isl.isl_multi_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_pw_aff_add.restype = c_void_p
isl.isl_multi_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_add_constant_multi_val.restype = c_void_p
isl.isl_multi_pw_aff_add_constant_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_add_constant_val.restype = c_void_p
isl.isl_multi_pw_aff_add_constant_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_bind.restype = c_void_p
isl.isl_multi_pw_aff_bind.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_bind_domain.restype = c_void_p
isl.isl_multi_pw_aff_bind_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_bind_domain_wrapped_domain.restype = c_void_p
isl.isl_multi_pw_aff_bind_domain_wrapped_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_coalesce.restype = c_void_p
isl.isl_multi_pw_aff_coalesce.argtypes = [c_void_p]
isl.isl_multi_pw_aff_domain.restype = c_void_p
isl.isl_multi_pw_aff_domain.argtypes = [c_void_p]
isl.isl_multi_pw_aff_flat_range_product.restype = c_void_p
isl.isl_multi_pw_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_get_at.restype = c_void_p
isl.isl_multi_pw_aff_get_at.argtypes = [c_void_p, c_int]
isl.isl_multi_pw_aff_get_list.restype = c_void_p
isl.isl_multi_pw_aff_get_list.argtypes = [c_void_p]
isl.isl_multi_pw_aff_get_space.restype = c_void_p
isl.isl_multi_pw_aff_get_space.argtypes = [c_void_p]
isl.isl_multi_pw_aff_gist.restype = c_void_p
isl.isl_multi_pw_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_identity_multi_pw_aff.restype = c_void_p
isl.isl_multi_pw_aff_identity_multi_pw_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_identity_on_domain_space.restype = c_void_p
isl.isl_multi_pw_aff_identity_on_domain_space.argtypes = [c_void_p]
isl.isl_multi_pw_aff_insert_domain.restype = c_void_p
isl.isl_multi_pw_aff_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_intersect_domain.restype = c_void_p
isl.isl_multi_pw_aff_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_intersect_params.restype = c_void_p
isl.isl_multi_pw_aff_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_involves_nan.argtypes = [c_void_p]
isl.isl_multi_pw_aff_involves_param_id.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_involves_param_id_list.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_max.restype = c_void_p
isl.isl_multi_pw_aff_max.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_max_multi_val.restype = c_void_p
isl.isl_multi_pw_aff_max_multi_val.argtypes = [c_void_p]
isl.isl_multi_pw_aff_min.restype = c_void_p
isl.isl_multi_pw_aff_min.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_min_multi_val.restype = c_void_p
isl.isl_multi_pw_aff_min_multi_val.argtypes = [c_void_p]
isl.isl_multi_pw_aff_neg.restype = c_void_p
isl.isl_multi_pw_aff_neg.argtypes = [c_void_p]
isl.isl_multi_pw_aff_plain_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_product.restype = c_void_p
isl.isl_multi_pw_aff_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_pullback_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_pullback_multi_pw_aff.restype = c_void_p
isl.isl_multi_pw_aff_pullback_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_pullback_pw_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_pullback_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_range_product.restype = c_void_p
isl.isl_multi_pw_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_scale_multi_val.restype = c_void_p
isl.isl_multi_pw_aff_scale_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_scale_val.restype = c_void_p
isl.isl_multi_pw_aff_scale_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_scale_down_multi_val.restype = c_void_p
isl.isl_multi_pw_aff_scale_down_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_scale_down_val.restype = c_void_p
isl.isl_multi_pw_aff_scale_down_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_set_at.restype = c_void_p
isl.isl_multi_pw_aff_set_at.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_multi_pw_aff_size.argtypes = [c_void_p]
isl.isl_multi_pw_aff_sub.restype = c_void_p
isl.isl_multi_pw_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_unbind_params_insert_domain.restype = c_void_p
isl.isl_multi_pw_aff_unbind_params_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_union_add.restype = c_void_p
isl.isl_multi_pw_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_zero.restype = c_void_p
isl.isl_multi_pw_aff_zero.argtypes = [c_void_p]
isl.isl_multi_pw_aff_copy.restype = c_void_p
isl.isl_multi_pw_aff_copy.argtypes = [c_void_p]
isl.isl_multi_pw_aff_free.restype = c_void_p
isl.isl_multi_pw_aff_free.argtypes = [c_void_p]
isl.isl_multi_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_multi_pw_aff_to_str.argtypes = [c_void_p]

class pw_multi_aff(union_pw_multi_aff, multi_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_from_multi_aff(isl.isl_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_from_pw_aff(isl.isl_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_pw_multi_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ptr = isl.isl_pw_multi_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.pw_multi_aff("""%s""")' % s
        else:
            return 'isl.pw_multi_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_add(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def add_constant(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_add_constant_multi_val(isl.isl_pw_multi_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_add_constant_val(isl.isl_pw_multi_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def as_multi_aff(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_as_multi_aff(isl.isl_pw_multi_aff_copy(arg0.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def bind_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_pw_multi_aff(arg0).bind_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_bind_domain(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def bind_domain_wrapped_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_pw_multi_aff(arg0).bind_domain_wrapped_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_bind_domain_wrapped_domain(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_coalesce(isl.isl_pw_multi_aff_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_domain(isl.isl_pw_multi_aff_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def domain_map(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_domain_map(isl.isl_space_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).flat_range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_flat_range_product(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def foreach_piece(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1, cb_arg2):
            cb_arg0 = set(ctx=arg0.ctx, ptr=(cb_arg0))
            cb_arg1 = multi_aff(ctx=arg0.ctx, ptr=(cb_arg1))
            try:
                arg1(cb_arg0, cb_arg1)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_foreach_piece(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def space(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_multi_aff(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_gist(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def identity_on_domain(*args):
        if len(args) == 1 and args[0].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_identity_on_domain_space(isl.isl_space_copy(args[0].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            return union_pw_multi_aff(arg0).insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_insert_domain(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_space_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_multi_aff(arg0).intersect_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_intersect_domain(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_multi_aff(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_intersect_params(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def involves_locals(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_involves_locals(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def isa_multi_aff(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_isa_multi_aff(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def max_multi_val(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_max_multi_val(isl.isl_pw_multi_aff_copy(arg0.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def min_multi_val(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_min_multi_val(isl.isl_pw_multi_aff_copy(arg0.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def n_piece(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_n_piece(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def preimage_domain_wrapped_domain(*args):
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_product(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_pullback_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_pullback_pw_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def range_factor_domain(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_range_factor_domain(isl.isl_pw_multi_aff_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def range_factor_range(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_range_factor_range(isl.isl_pw_multi_aff_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def range_map(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_range_map(isl.isl_space_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_range_product(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def scale(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_scale_val(isl.isl_pw_multi_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale_down(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_multi_aff_scale_down_val(isl.isl_pw_multi_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = pw_multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).sub(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_sub(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def subtract_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_multi_aff(arg0).subtract_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_subtract_domain(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).union_add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_union_add(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def zero(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_zero(isl.isl_space_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj

isl.isl_pw_multi_aff_from_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_from_multi_aff.argtypes = [c_void_p]
isl.isl_pw_multi_aff_from_pw_aff.restype = c_void_p
isl.isl_pw_multi_aff_from_pw_aff.argtypes = [c_void_p]
isl.isl_pw_multi_aff_read_from_str.restype = c_void_p
isl.isl_pw_multi_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_pw_multi_aff_add.restype = c_void_p
isl.isl_pw_multi_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_add_constant_multi_val.restype = c_void_p
isl.isl_pw_multi_aff_add_constant_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_add_constant_val.restype = c_void_p
isl.isl_pw_multi_aff_add_constant_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_as_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_as_multi_aff.argtypes = [c_void_p]
isl.isl_pw_multi_aff_bind_domain.restype = c_void_p
isl.isl_pw_multi_aff_bind_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_bind_domain_wrapped_domain.restype = c_void_p
isl.isl_pw_multi_aff_bind_domain_wrapped_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_coalesce.restype = c_void_p
isl.isl_pw_multi_aff_coalesce.argtypes = [c_void_p]
isl.isl_pw_multi_aff_domain.restype = c_void_p
isl.isl_pw_multi_aff_domain.argtypes = [c_void_p]
isl.isl_pw_multi_aff_domain_map.restype = c_void_p
isl.isl_pw_multi_aff_domain_map.argtypes = [c_void_p]
isl.isl_pw_multi_aff_flat_range_product.restype = c_void_p
isl.isl_pw_multi_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_foreach_piece.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_pw_multi_aff_get_space.restype = c_void_p
isl.isl_pw_multi_aff_get_space.argtypes = [c_void_p]
isl.isl_pw_multi_aff_gist.restype = c_void_p
isl.isl_pw_multi_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_identity_on_domain_space.restype = c_void_p
isl.isl_pw_multi_aff_identity_on_domain_space.argtypes = [c_void_p]
isl.isl_pw_multi_aff_insert_domain.restype = c_void_p
isl.isl_pw_multi_aff_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_intersect_domain.restype = c_void_p
isl.isl_pw_multi_aff_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_intersect_params.restype = c_void_p
isl.isl_pw_multi_aff_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_involves_locals.argtypes = [c_void_p]
isl.isl_pw_multi_aff_isa_multi_aff.argtypes = [c_void_p]
isl.isl_pw_multi_aff_max_multi_val.restype = c_void_p
isl.isl_pw_multi_aff_max_multi_val.argtypes = [c_void_p]
isl.isl_pw_multi_aff_min_multi_val.restype = c_void_p
isl.isl_pw_multi_aff_min_multi_val.argtypes = [c_void_p]
isl.isl_pw_multi_aff_n_piece.argtypes = [c_void_p]
isl.isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_product.restype = c_void_p
isl.isl_pw_multi_aff_product.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_pullback_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_pullback_pw_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_pullback_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_range_factor_domain.restype = c_void_p
isl.isl_pw_multi_aff_range_factor_domain.argtypes = [c_void_p]
isl.isl_pw_multi_aff_range_factor_range.restype = c_void_p
isl.isl_pw_multi_aff_range_factor_range.argtypes = [c_void_p]
isl.isl_pw_multi_aff_range_map.restype = c_void_p
isl.isl_pw_multi_aff_range_map.argtypes = [c_void_p]
isl.isl_pw_multi_aff_range_product.restype = c_void_p
isl.isl_pw_multi_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_scale_val.restype = c_void_p
isl.isl_pw_multi_aff_scale_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_scale_down_val.restype = c_void_p
isl.isl_pw_multi_aff_scale_down_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_sub.restype = c_void_p
isl.isl_pw_multi_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_subtract_domain.restype = c_void_p
isl.isl_pw_multi_aff_subtract_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_union_add.restype = c_void_p
isl.isl_pw_multi_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_zero.restype = c_void_p
isl.isl_pw_multi_aff_zero.argtypes = [c_void_p]
isl.isl_pw_multi_aff_copy.restype = c_void_p
isl.isl_pw_multi_aff_copy.argtypes = [c_void_p]
isl.isl_pw_multi_aff_free.restype = c_void_p
isl.isl_pw_multi_aff_free.argtypes = [c_void_p]
isl.isl_pw_multi_aff_to_str.restype = POINTER(c_char)
isl.isl_pw_multi_aff_to_str.argtypes = [c_void_p]

class pw_aff(union_pw_aff, pw_multi_aff, multi_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_aff_from_aff(isl.isl_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_pw_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.pw_aff("""%s""")' % s
        else:
            return 'isl.pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_add(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def add_constant(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_aff_add_constant_val(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def as_aff(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_as_aff(isl.isl_pw_aff_copy(arg0.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def bind(*args):
        if len(args) == 2 and (args[1].__class__ is id or type(args[1]) == str):
            args = list(args)
            try:
                if not args[1].__class__ is id:
                    args[1] = id(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_aff_bind_id(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_id_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def bind_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_pw_aff(arg0).bind_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_bind_domain(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def bind_domain_wrapped_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_pw_aff(arg0).bind_domain_wrapped_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_bind_domain_wrapped_domain(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def ceil(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_ceil(isl.isl_pw_aff_copy(arg0.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_coalesce(isl.isl_pw_aff_copy(arg0.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def cond(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).cond(arg1, arg2)
        try:
            if not arg2.__class__ is pw_aff:
                arg2 = pw_aff(arg2)
        except:
            return union_pw_aff(arg0).cond(arg1, arg2)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_cond(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr), isl.isl_pw_aff_copy(arg2.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def div(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).div(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_div(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_domain(isl.isl_pw_aff_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def eq_set(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).eq_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_eq_set(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def eval(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is point:
                arg1 = point(arg1)
        except:
            return union_pw_aff(arg0).eval(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_eval(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_point_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def floor(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_floor(isl.isl_pw_aff_copy(arg0.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def ge_set(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).ge_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_ge_set(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_aff(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_gist(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def gt_set(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).gt_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_gt_set(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            return union_pw_aff(arg0).insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_insert_domain(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_space_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_aff(arg0).intersect_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_intersect_domain(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_aff(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_intersect_params(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def isa_aff(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_isa_aff(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def le_set(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).le_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_le_set(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def lt_set(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).lt_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_lt_set(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def max(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).max(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_max(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def min(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).min(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_min(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def mod(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_aff_mod_val(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def mul(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).mul(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_mul(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def ne_set(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).ne_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_ne_set(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def neg(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_neg(isl.isl_pw_aff_copy(arg0.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def param_on_domain(*args):
        if len(args) == 2 and args[0].__class__ is set and (args[1].__class__ is id or type(args[1]) == str):
            args = list(args)
            try:
                if not args[1].__class__ is id:
                    args[1] = id(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_aff_param_on_domain_id(isl.isl_set_copy(args[0].ptr), isl.isl_id_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_pw_aff_pullback_multi_aff(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_pw_aff_pullback_multi_pw_aff(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_pw_aff_pullback_pw_multi_aff(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_aff_scale_val(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale_down(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_pw_aff_scale_down_val(isl.isl_pw_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = pw_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).sub(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_sub(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def subtract_domain(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_pw_aff(arg0).subtract_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_subtract_domain(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def tdiv_q(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).tdiv_q(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_tdiv_q(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def tdiv_r(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).tdiv_r(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_tdiv_r(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).union_add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_union_add(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj

isl.isl_pw_aff_from_aff.restype = c_void_p
isl.isl_pw_aff_from_aff.argtypes = [c_void_p]
isl.isl_pw_aff_read_from_str.restype = c_void_p
isl.isl_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_pw_aff_add.restype = c_void_p
isl.isl_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_add_constant_val.restype = c_void_p
isl.isl_pw_aff_add_constant_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_as_aff.restype = c_void_p
isl.isl_pw_aff_as_aff.argtypes = [c_void_p]
isl.isl_pw_aff_bind_id.restype = c_void_p
isl.isl_pw_aff_bind_id.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_bind_domain.restype = c_void_p
isl.isl_pw_aff_bind_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_bind_domain_wrapped_domain.restype = c_void_p
isl.isl_pw_aff_bind_domain_wrapped_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_ceil.restype = c_void_p
isl.isl_pw_aff_ceil.argtypes = [c_void_p]
isl.isl_pw_aff_coalesce.restype = c_void_p
isl.isl_pw_aff_coalesce.argtypes = [c_void_p]
isl.isl_pw_aff_cond.restype = c_void_p
isl.isl_pw_aff_cond.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_pw_aff_div.restype = c_void_p
isl.isl_pw_aff_div.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_domain.restype = c_void_p
isl.isl_pw_aff_domain.argtypes = [c_void_p]
isl.isl_pw_aff_eq_set.restype = c_void_p
isl.isl_pw_aff_eq_set.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_eval.restype = c_void_p
isl.isl_pw_aff_eval.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_floor.restype = c_void_p
isl.isl_pw_aff_floor.argtypes = [c_void_p]
isl.isl_pw_aff_ge_set.restype = c_void_p
isl.isl_pw_aff_ge_set.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_gist.restype = c_void_p
isl.isl_pw_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_gt_set.restype = c_void_p
isl.isl_pw_aff_gt_set.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_insert_domain.restype = c_void_p
isl.isl_pw_aff_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_intersect_domain.restype = c_void_p
isl.isl_pw_aff_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_intersect_params.restype = c_void_p
isl.isl_pw_aff_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_isa_aff.argtypes = [c_void_p]
isl.isl_pw_aff_le_set.restype = c_void_p
isl.isl_pw_aff_le_set.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_lt_set.restype = c_void_p
isl.isl_pw_aff_lt_set.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_max.restype = c_void_p
isl.isl_pw_aff_max.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_min.restype = c_void_p
isl.isl_pw_aff_min.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_mod_val.restype = c_void_p
isl.isl_pw_aff_mod_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_mul.restype = c_void_p
isl.isl_pw_aff_mul.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_ne_set.restype = c_void_p
isl.isl_pw_aff_ne_set.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_neg.restype = c_void_p
isl.isl_pw_aff_neg.argtypes = [c_void_p]
isl.isl_pw_aff_param_on_domain_id.restype = c_void_p
isl.isl_pw_aff_param_on_domain_id.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_pullback_multi_aff.restype = c_void_p
isl.isl_pw_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_pullback_multi_pw_aff.restype = c_void_p
isl.isl_pw_aff_pullback_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_pullback_pw_multi_aff.restype = c_void_p
isl.isl_pw_aff_pullback_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_scale_val.restype = c_void_p
isl.isl_pw_aff_scale_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_scale_down_val.restype = c_void_p
isl.isl_pw_aff_scale_down_val.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_sub.restype = c_void_p
isl.isl_pw_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_subtract_domain.restype = c_void_p
isl.isl_pw_aff_subtract_domain.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_tdiv_q.restype = c_void_p
isl.isl_pw_aff_tdiv_q.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_tdiv_r.restype = c_void_p
isl.isl_pw_aff_tdiv_r.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_union_add.restype = c_void_p
isl.isl_pw_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_copy.restype = c_void_p
isl.isl_pw_aff_copy.argtypes = [c_void_p]
isl.isl_pw_aff_free.restype = c_void_p
isl.isl_pw_aff_free.argtypes = [c_void_p]
isl.isl_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_pw_aff_to_str.argtypes = [c_void_p]

class multi_aff(pw_multi_aff, multi_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_aff_from_aff(isl.isl_aff_copy(args[0].ptr))
            return
        if len(args) == 2 and args[0].__class__ is space and args[1].__class__ is aff_list:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_aff_from_aff_list(isl.isl_space_copy(args[0].ptr), isl.isl_aff_list_copy(args[1].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ptr = isl.isl_multi_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_aff("""%s""")' % s
        else:
            return 'isl.multi_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_add(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def add_constant(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_aff_add_constant_multi_val(isl.isl_multi_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_aff_add_constant_val(isl.isl_multi_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def bind(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return pw_multi_aff(arg0).bind(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_bind(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def bind_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return pw_multi_aff(arg0).bind_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_bind_domain(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def bind_domain_wrapped_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return pw_multi_aff(arg0).bind_domain_wrapped_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_bind_domain_wrapped_domain(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def domain_map(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_domain_map(isl.isl_space_copy(arg0.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).flat_range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_flat_range_product(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def floor(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_floor(isl.isl_multi_aff_copy(arg0.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_get_at(arg0.ptr, arg1)
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def constant_multi_val(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_get_constant_multi_val(arg0.ptr)
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def get_constant_multi_val(arg0):
        return arg0.constant_multi_val()
    def list(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_get_list(arg0.ptr)
        obj = aff_list(ctx=ctx, ptr=res)
        return obj
    def get_list(arg0):
        return arg0.list()
    def space(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return pw_multi_aff(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_gist(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def identity(*args):
        if len(args) == 1:
            ctx = args[0].ctx
            res = isl.isl_multi_aff_identity_multi_aff(isl.isl_multi_aff_copy(args[0].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    @staticmethod
    def identity_on_domain(*args):
        if len(args) == 1 and args[0].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_multi_aff_identity_on_domain_space(isl.isl_space_copy(args[0].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            return pw_multi_aff(arg0).insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_insert_domain(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_space_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def involves_locals(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_involves_locals(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def involves_nan(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_involves_nan(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def neg(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_neg(isl.isl_multi_aff_copy(arg0.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def plain_is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).plain_is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_plain_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_product(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_multi_aff_pullback_multi_aff(isl.isl_multi_aff_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    @staticmethod
    def range_map(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_range_map(isl.isl_space_copy(arg0.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_range_product(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def scale(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_aff_scale_multi_val(isl.isl_multi_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_aff_scale_val(isl.isl_multi_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale_down(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_aff_scale_down_multi_val(isl.isl_multi_aff_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_aff_scale_down_val(isl.isl_multi_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def set_at(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is aff:
                arg2 = aff(arg2)
        except:
            return pw_multi_aff(arg0).set_at(arg1, arg2)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_set_at(isl.isl_multi_aff_copy(arg0.ptr), arg1, isl.isl_aff_copy(arg2.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).sub(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_sub(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def unbind_params_insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return pw_multi_aff(arg0).unbind_params_insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_unbind_params_insert_domain(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def zero(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_aff_zero(isl.isl_space_copy(arg0.ptr))
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj

isl.isl_multi_aff_from_aff.restype = c_void_p
isl.isl_multi_aff_from_aff.argtypes = [c_void_p]
isl.isl_multi_aff_from_aff_list.restype = c_void_p
isl.isl_multi_aff_from_aff_list.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_read_from_str.restype = c_void_p
isl.isl_multi_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_aff_add.restype = c_void_p
isl.isl_multi_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_add_constant_multi_val.restype = c_void_p
isl.isl_multi_aff_add_constant_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_add_constant_val.restype = c_void_p
isl.isl_multi_aff_add_constant_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_bind.restype = c_void_p
isl.isl_multi_aff_bind.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_bind_domain.restype = c_void_p
isl.isl_multi_aff_bind_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_bind_domain_wrapped_domain.restype = c_void_p
isl.isl_multi_aff_bind_domain_wrapped_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_domain_map.restype = c_void_p
isl.isl_multi_aff_domain_map.argtypes = [c_void_p]
isl.isl_multi_aff_flat_range_product.restype = c_void_p
isl.isl_multi_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_floor.restype = c_void_p
isl.isl_multi_aff_floor.argtypes = [c_void_p]
isl.isl_multi_aff_get_at.restype = c_void_p
isl.isl_multi_aff_get_at.argtypes = [c_void_p, c_int]
isl.isl_multi_aff_get_constant_multi_val.restype = c_void_p
isl.isl_multi_aff_get_constant_multi_val.argtypes = [c_void_p]
isl.isl_multi_aff_get_list.restype = c_void_p
isl.isl_multi_aff_get_list.argtypes = [c_void_p]
isl.isl_multi_aff_get_space.restype = c_void_p
isl.isl_multi_aff_get_space.argtypes = [c_void_p]
isl.isl_multi_aff_gist.restype = c_void_p
isl.isl_multi_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_identity_multi_aff.restype = c_void_p
isl.isl_multi_aff_identity_multi_aff.argtypes = [c_void_p]
isl.isl_multi_aff_identity_on_domain_space.restype = c_void_p
isl.isl_multi_aff_identity_on_domain_space.argtypes = [c_void_p]
isl.isl_multi_aff_insert_domain.restype = c_void_p
isl.isl_multi_aff_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_involves_locals.argtypes = [c_void_p]
isl.isl_multi_aff_involves_nan.argtypes = [c_void_p]
isl.isl_multi_aff_neg.restype = c_void_p
isl.isl_multi_aff_neg.argtypes = [c_void_p]
isl.isl_multi_aff_plain_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_product.restype = c_void_p
isl.isl_multi_aff_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_pullback_multi_aff.restype = c_void_p
isl.isl_multi_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_range_map.restype = c_void_p
isl.isl_multi_aff_range_map.argtypes = [c_void_p]
isl.isl_multi_aff_range_product.restype = c_void_p
isl.isl_multi_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_scale_multi_val.restype = c_void_p
isl.isl_multi_aff_scale_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_scale_val.restype = c_void_p
isl.isl_multi_aff_scale_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_scale_down_multi_val.restype = c_void_p
isl.isl_multi_aff_scale_down_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_scale_down_val.restype = c_void_p
isl.isl_multi_aff_scale_down_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_set_at.restype = c_void_p
isl.isl_multi_aff_set_at.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_multi_aff_size.argtypes = [c_void_p]
isl.isl_multi_aff_sub.restype = c_void_p
isl.isl_multi_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_unbind_params_insert_domain.restype = c_void_p
isl.isl_multi_aff_unbind_params_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_zero.restype = c_void_p
isl.isl_multi_aff_zero.argtypes = [c_void_p]
isl.isl_multi_aff_copy.restype = c_void_p
isl.isl_multi_aff_copy.argtypes = [c_void_p]
isl.isl_multi_aff_free.restype = c_void_p
isl.isl_multi_aff_free.argtypes = [c_void_p]
isl.isl_multi_aff_to_str.restype = POINTER(c_char)
isl.isl_multi_aff_to_str.argtypes = [c_void_p]

class aff(pw_aff, multi_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_aff_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        ptr = isl.isl_aff_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.aff("""%s""")' % s
        else:
            return 'isl.aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_add(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def add_constant(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_aff_add_constant_val(isl.isl_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def bind(*args):
        if len(args) == 2 and (args[1].__class__ is id or type(args[1]) == str):
            args = list(args)
            try:
                if not args[1].__class__ is id:
                    args[1] = id(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_aff_bind_id(isl.isl_aff_copy(args[0].ptr), isl.isl_id_copy(args[1].ptr))
            obj = basic_set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def ceil(arg0):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_ceil(isl.isl_aff_copy(arg0.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def div(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).div(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_div(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def eq_set(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).eq_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_eq_set(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def eval(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is point:
                arg1 = point(arg1)
        except:
            return pw_aff(arg0).eval(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_eval(isl.isl_aff_copy(arg0.ptr), isl.isl_point_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def floor(arg0):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_floor(isl.isl_aff_copy(arg0.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def ge_set(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).ge_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_ge_set(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def constant_val(arg0):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_get_constant_val(arg0.ptr)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def get_constant_val(arg0):
        return arg0.constant_val()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return pw_aff(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_gist(isl.isl_aff_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def gt_set(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).gt_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_gt_set(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def is_cst(arg0):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_is_cst(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def le_set(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).le_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_le_set(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def lt_set(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).lt_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_lt_set(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def mod(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_aff_mod_val(isl.isl_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def mul(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).mul(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_mul(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def ne_set(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).ne_set(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_ne_set(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def neg(arg0):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_neg(isl.isl_aff_copy(arg0.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_aff_pullback_multi_aff(isl.isl_aff_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_aff_scale_val(isl.isl_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale_down(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_aff_scale_down_val(isl.isl_aff_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = aff(ctx=ctx, ptr=res)
            return obj
        raise Error
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).sub(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_sub(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def unbind_params_insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return pw_aff(arg0).unbind_params_insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_unbind_params_insert_domain(isl.isl_aff_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = aff(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def zero_on_domain(*args):
        if len(args) == 1 and args[0].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_aff_zero_on_domain_space(isl.isl_space_copy(args[0].ptr))
            obj = aff(ctx=ctx, ptr=res)
            return obj
        raise Error

isl.isl_aff_read_from_str.restype = c_void_p
isl.isl_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_aff_add.restype = c_void_p
isl.isl_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_aff_add_constant_val.restype = c_void_p
isl.isl_aff_add_constant_val.argtypes = [c_void_p, c_void_p]
isl.isl_aff_bind_id.restype = c_void_p
isl.isl_aff_bind_id.argtypes = [c_void_p, c_void_p]
isl.isl_aff_ceil.restype = c_void_p
isl.isl_aff_ceil.argtypes = [c_void_p]
isl.isl_aff_div.restype = c_void_p
isl.isl_aff_div.argtypes = [c_void_p, c_void_p]
isl.isl_aff_eq_set.restype = c_void_p
isl.isl_aff_eq_set.argtypes = [c_void_p, c_void_p]
isl.isl_aff_eval.restype = c_void_p
isl.isl_aff_eval.argtypes = [c_void_p, c_void_p]
isl.isl_aff_floor.restype = c_void_p
isl.isl_aff_floor.argtypes = [c_void_p]
isl.isl_aff_ge_set.restype = c_void_p
isl.isl_aff_ge_set.argtypes = [c_void_p, c_void_p]
isl.isl_aff_get_constant_val.restype = c_void_p
isl.isl_aff_get_constant_val.argtypes = [c_void_p]
isl.isl_aff_gist.restype = c_void_p
isl.isl_aff_gist.argtypes = [c_void_p, c_void_p]
isl.isl_aff_gt_set.restype = c_void_p
isl.isl_aff_gt_set.argtypes = [c_void_p, c_void_p]
isl.isl_aff_is_cst.argtypes = [c_void_p]
isl.isl_aff_le_set.restype = c_void_p
isl.isl_aff_le_set.argtypes = [c_void_p, c_void_p]
isl.isl_aff_lt_set.restype = c_void_p
isl.isl_aff_lt_set.argtypes = [c_void_p, c_void_p]
isl.isl_aff_mod_val.restype = c_void_p
isl.isl_aff_mod_val.argtypes = [c_void_p, c_void_p]
isl.isl_aff_mul.restype = c_void_p
isl.isl_aff_mul.argtypes = [c_void_p, c_void_p]
isl.isl_aff_ne_set.restype = c_void_p
isl.isl_aff_ne_set.argtypes = [c_void_p, c_void_p]
isl.isl_aff_neg.restype = c_void_p
isl.isl_aff_neg.argtypes = [c_void_p]
isl.isl_aff_pullback_multi_aff.restype = c_void_p
isl.isl_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_aff_scale_val.restype = c_void_p
isl.isl_aff_scale_val.argtypes = [c_void_p, c_void_p]
isl.isl_aff_scale_down_val.restype = c_void_p
isl.isl_aff_scale_down_val.argtypes = [c_void_p, c_void_p]
isl.isl_aff_sub.restype = c_void_p
isl.isl_aff_sub.argtypes = [c_void_p, c_void_p]
isl.isl_aff_unbind_params_insert_domain.restype = c_void_p
isl.isl_aff_unbind_params_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_aff_zero_on_domain_space.restype = c_void_p
isl.isl_aff_zero_on_domain_space.argtypes = [c_void_p]
isl.isl_aff_copy.restype = c_void_p
isl.isl_aff_copy.argtypes = [c_void_p]
isl.isl_aff_free.restype = c_void_p
isl.isl_aff_free.argtypes = [c_void_p]
isl.isl_aff_to_str.restype = POINTER(c_char)
isl.isl_aff_to_str.argtypes = [c_void_p]

class aff_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_aff_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_aff_list_from_aff(isl.isl_aff_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_aff_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        ptr = isl.isl_aff_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.aff_list("""%s""")' % s
        else:
            return 'isl.aff_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_list_add(isl.isl_aff_list_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        obj = aff_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_list_clear(isl.isl_aff_list_copy(arg0.ptr))
        obj = aff_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff_list:
                arg1 = aff_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_list_concat(isl.isl_aff_list_copy(arg0.ptr), isl.isl_aff_list_copy(arg1.ptr))
        obj = aff_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_list_drop(isl.isl_aff_list_copy(arg0.ptr), arg1, arg2)
        obj = aff_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = aff(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_aff_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_list_get_at(arg0.ptr, arg1)
        obj = aff(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is aff:
                arg2 = aff(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_list_insert(isl.isl_aff_list_copy(arg0.ptr), arg1, isl.isl_aff_copy(arg2.ptr))
        obj = aff_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is aff_list:
                arg0 = aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_aff_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_aff_list_alloc.restype = c_void_p
isl.isl_aff_list_alloc.argtypes = [Context, c_int]
isl.isl_aff_list_from_aff.restype = c_void_p
isl.isl_aff_list_from_aff.argtypes = [c_void_p]
isl.isl_aff_list_add.restype = c_void_p
isl.isl_aff_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_aff_list_clear.restype = c_void_p
isl.isl_aff_list_clear.argtypes = [c_void_p]
isl.isl_aff_list_concat.restype = c_void_p
isl.isl_aff_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_aff_list_drop.restype = c_void_p
isl.isl_aff_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_aff_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_aff_list_get_at.restype = c_void_p
isl.isl_aff_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_aff_list_insert.restype = c_void_p
isl.isl_aff_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_aff_list_size.argtypes = [c_void_p]
isl.isl_aff_list_copy.restype = c_void_p
isl.isl_aff_list_copy.argtypes = [c_void_p]
isl.isl_aff_list_free.restype = c_void_p
isl.isl_aff_list_free.argtypes = [c_void_p]
isl.isl_aff_list_to_str.restype = POINTER(c_char)
isl.isl_aff_list_to_str.argtypes = [c_void_p]

class ast_build(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 0:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_ast_build_alloc(self.ctx)
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_build_free(self.ptr)
    def copy_callbacks(self, obj):
        if hasattr(obj, 'at_each_domain'):
            self.at_each_domain = obj.at_each_domain
    def set_at_each_domain(arg0, arg1):
        try:
            if not arg0.__class__ is ast_build:
                arg0 = ast_build(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_void_p, c_void_p, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1, cb_arg2):
            cb_arg0 = ast_node(ctx=arg0.ctx, ptr=(cb_arg0))
            cb_arg1 = ast_build(ctx=arg0.ctx, ptr=isl.isl_ast_build_copy(cb_arg1))
            try:
                res = arg1(cb_arg0, cb_arg1)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return None
            return isl.isl_ast_node_copy(res.ptr)
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_ast_build_set_at_each_domain(isl.isl_ast_build_copy(arg0.ptr), cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if hasattr(arg0, 'at_each_domain') and arg0.at_each_domain['exc_info'] != None:
            exc_info = arg0.at_each_domain['exc_info'][0]
            arg0.at_each_domain['exc_info'][0] = None
            if exc_info != None:
                raise (exc_info[0], exc_info[1], exc_info[2])
        obj = ast_build(ctx=ctx, ptr=res)
        obj.copy_callbacks(arg0)
        obj.at_each_domain = { 'func': cb, 'exc_info': exc_info }
        return obj
    def access_from(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_ast_build_access_from_multi_pw_aff(args[0].ptr, isl.isl_multi_pw_aff_copy(args[1].ptr))
            if hasattr(args[0], 'at_each_domain') and args[0].at_each_domain['exc_info'] != None:
                exc_info = args[0].at_each_domain['exc_info'][0]
                args[0].at_each_domain['exc_info'][0] = None
                if exc_info != None:
                    raise (exc_info[0], exc_info[1], exc_info[2])
            obj = ast_expr(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_ast_build_access_from_pw_multi_aff(args[0].ptr, isl.isl_pw_multi_aff_copy(args[1].ptr))
            if hasattr(args[0], 'at_each_domain') and args[0].at_each_domain['exc_info'] != None:
                exc_info = args[0].at_each_domain['exc_info'][0]
                args[0].at_each_domain['exc_info'][0] = None
                if exc_info != None:
                    raise (exc_info[0], exc_info[1], exc_info[2])
            obj = ast_expr(ctx=ctx, ptr=res)
            return obj
        raise Error
    def call_from(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_ast_build_call_from_multi_pw_aff(args[0].ptr, isl.isl_multi_pw_aff_copy(args[1].ptr))
            if hasattr(args[0], 'at_each_domain') and args[0].at_each_domain['exc_info'] != None:
                exc_info = args[0].at_each_domain['exc_info'][0]
                args[0].at_each_domain['exc_info'][0] = None
                if exc_info != None:
                    raise (exc_info[0], exc_info[1], exc_info[2])
            obj = ast_expr(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_ast_build_call_from_pw_multi_aff(args[0].ptr, isl.isl_pw_multi_aff_copy(args[1].ptr))
            if hasattr(args[0], 'at_each_domain') and args[0].at_each_domain['exc_info'] != None:
                exc_info = args[0].at_each_domain['exc_info'][0]
                args[0].at_each_domain['exc_info'][0] = None
                if exc_info != None:
                    raise (exc_info[0], exc_info[1], exc_info[2])
            obj = ast_expr(ctx=ctx, ptr=res)
            return obj
        raise Error
    def expr_from(*args):
        if len(args) == 2 and args[1].__class__ is pw_aff:
            ctx = args[0].ctx
            res = isl.isl_ast_build_expr_from_pw_aff(args[0].ptr, isl.isl_pw_aff_copy(args[1].ptr))
            if hasattr(args[0], 'at_each_domain') and args[0].at_each_domain['exc_info'] != None:
                exc_info = args[0].at_each_domain['exc_info'][0]
                args[0].at_each_domain['exc_info'][0] = None
                if exc_info != None:
                    raise (exc_info[0], exc_info[1], exc_info[2])
            obj = ast_expr(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is set:
            ctx = args[0].ctx
            res = isl.isl_ast_build_expr_from_set(args[0].ptr, isl.isl_set_copy(args[1].ptr))
            if hasattr(args[0], 'at_each_domain') and args[0].at_each_domain['exc_info'] != None:
                exc_info = args[0].at_each_domain['exc_info'][0]
                args[0].at_each_domain['exc_info'][0] = None
                if exc_info != None:
                    raise (exc_info[0], exc_info[1], exc_info[2])
            obj = ast_expr(ctx=ctx, ptr=res)
            return obj
        raise Error
    @staticmethod
    def from_context(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_build_from_context(isl.isl_set_copy(arg0.ptr))
        obj = ast_build(ctx=ctx, ptr=res)
        return obj
    def schedule(arg0):
        try:
            if not arg0.__class__ is ast_build:
                arg0 = ast_build(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_build_get_schedule(arg0.ptr)
        if hasattr(arg0, 'at_each_domain') and arg0.at_each_domain['exc_info'] != None:
            exc_info = arg0.at_each_domain['exc_info'][0]
            arg0.at_each_domain['exc_info'][0] = None
            if exc_info != None:
                raise (exc_info[0], exc_info[1], exc_info[2])
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_schedule(arg0):
        return arg0.schedule()
    def node_from(*args):
        if len(args) == 2 and args[1].__class__ is schedule:
            ctx = args[0].ctx
            res = isl.isl_ast_build_node_from_schedule(args[0].ptr, isl.isl_schedule_copy(args[1].ptr))
            if hasattr(args[0], 'at_each_domain') and args[0].at_each_domain['exc_info'] != None:
                exc_info = args[0].at_each_domain['exc_info'][0]
                args[0].at_each_domain['exc_info'][0] = None
                if exc_info != None:
                    raise (exc_info[0], exc_info[1], exc_info[2])
            obj = ast_node(ctx=ctx, ptr=res)
            return obj
        raise Error
    def node_from_schedule_map(arg0, arg1):
        try:
            if not arg0.__class__ is ast_build:
                arg0 = ast_build(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_build_node_from_schedule_map(arg0.ptr, isl.isl_union_map_copy(arg1.ptr))
        if hasattr(arg0, 'at_each_domain') and arg0.at_each_domain['exc_info'] != None:
            exc_info = arg0.at_each_domain['exc_info'][0]
            arg0.at_each_domain['exc_info'][0] = None
            if exc_info != None:
                raise (exc_info[0], exc_info[1], exc_info[2])
        obj = ast_node(ctx=ctx, ptr=res)
        return obj

isl.isl_ast_build_alloc.restype = c_void_p
isl.isl_ast_build_alloc.argtypes = [Context]
isl.isl_ast_build_set_at_each_domain.restype = c_void_p
isl.isl_ast_build_set_at_each_domain.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_ast_build_access_from_multi_pw_aff.restype = c_void_p
isl.isl_ast_build_access_from_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_access_from_pw_multi_aff.restype = c_void_p
isl.isl_ast_build_access_from_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_call_from_multi_pw_aff.restype = c_void_p
isl.isl_ast_build_call_from_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_call_from_pw_multi_aff.restype = c_void_p
isl.isl_ast_build_call_from_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_expr_from_pw_aff.restype = c_void_p
isl.isl_ast_build_expr_from_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_expr_from_set.restype = c_void_p
isl.isl_ast_build_expr_from_set.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_from_context.restype = c_void_p
isl.isl_ast_build_from_context.argtypes = [c_void_p]
isl.isl_ast_build_get_schedule.restype = c_void_p
isl.isl_ast_build_get_schedule.argtypes = [c_void_p]
isl.isl_ast_build_node_from_schedule.restype = c_void_p
isl.isl_ast_build_node_from_schedule.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_node_from_schedule_map.restype = c_void_p
isl.isl_ast_build_node_from_schedule_map.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_copy.restype = c_void_p
isl.isl_ast_build_copy.argtypes = [c_void_p]
isl.isl_ast_build_free.restype = c_void_p
isl.isl_ast_build_free.argtypes = [c_void_p]

class ast_expr(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_id):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_int):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        if "ptr" in keywords:
            type = isl.isl_ast_expr_get_type(keywords["ptr"])
            if type == 0:
                return ast_expr_op(**keywords)
            if type == 1:
                return ast_expr_id(**keywords)
            if type == 2:
                return ast_expr_int(**keywords)
            raise
        return super(ast_expr, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr("""%s""")' % s
        else:
            return 'isl.ast_expr("%s")' % s
    def to_C_str(arg0):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_expr_to_C_str(arg0.ptr)
        if res == 0:
            raise
        string = cast(res, c_char_p).value.decode('ascii')
        libc.free(res)
        return string

isl.isl_ast_expr_to_C_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_C_str.argtypes = [c_void_p]
isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]
isl.isl_ast_expr_get_type.argtypes = [c_void_p]

class ast_expr_id(ast_expr):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_id, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_id:
                arg0 = ast_expr_id(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_id("""%s""")' % s
        else:
            return 'isl.ast_expr_id("%s")' % s
    def id(arg0):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_expr_id_get_id(arg0.ptr)
        obj = id(ctx=ctx, ptr=res)
        return obj
    def get_id(arg0):
        return arg0.id()

isl.isl_ast_expr_id_get_id.restype = c_void_p
isl.isl_ast_expr_id_get_id.argtypes = [c_void_p]
isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_int(ast_expr):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_int, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_int:
                arg0 = ast_expr_int(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_int("""%s""")' % s
        else:
            return 'isl.ast_expr_int("%s")' % s
    def val(arg0):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_expr_int_get_val(arg0.ptr)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def get_val(arg0):
        return arg0.val()

isl.isl_ast_expr_int_get_val.restype = c_void_p
isl.isl_ast_expr_int_get_val.argtypes = [c_void_p]
isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op(ast_expr):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_and):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_and_then):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_or):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_or_else):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_max):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_min):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_minus):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_add):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_sub):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_mul):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_div):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_fdiv_q):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_pdiv_q):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_pdiv_r):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_zdiv_r):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_cond):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_select):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_eq):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_le):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_lt):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_ge):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_gt):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_call):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_access):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_member):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_expr_op_address_of):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_expr_copy(args[0].ptr)
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        if "ptr" in keywords:
            type = isl.isl_ast_expr_op_get_type(keywords["ptr"])
            if type == 0:
                return ast_expr_op_and(**keywords)
            if type == 1:
                return ast_expr_op_and_then(**keywords)
            if type == 2:
                return ast_expr_op_or(**keywords)
            if type == 3:
                return ast_expr_op_or_else(**keywords)
            if type == 4:
                return ast_expr_op_max(**keywords)
            if type == 5:
                return ast_expr_op_min(**keywords)
            if type == 6:
                return ast_expr_op_minus(**keywords)
            if type == 7:
                return ast_expr_op_add(**keywords)
            if type == 8:
                return ast_expr_op_sub(**keywords)
            if type == 9:
                return ast_expr_op_mul(**keywords)
            if type == 10:
                return ast_expr_op_div(**keywords)
            if type == 11:
                return ast_expr_op_fdiv_q(**keywords)
            if type == 12:
                return ast_expr_op_pdiv_q(**keywords)
            if type == 13:
                return ast_expr_op_pdiv_r(**keywords)
            if type == 14:
                return ast_expr_op_zdiv_r(**keywords)
            if type == 15:
                return ast_expr_op_cond(**keywords)
            if type == 16:
                return ast_expr_op_select(**keywords)
            if type == 17:
                return ast_expr_op_eq(**keywords)
            if type == 18:
                return ast_expr_op_le(**keywords)
            if type == 19:
                return ast_expr_op_lt(**keywords)
            if type == 20:
                return ast_expr_op_ge(**keywords)
            if type == 21:
                return ast_expr_op_gt(**keywords)
            if type == 22:
                return ast_expr_op_call(**keywords)
            if type == 23:
                return ast_expr_op_access(**keywords)
            if type == 24:
                return ast_expr_op_member(**keywords)
            if type == 25:
                return ast_expr_op_address_of(**keywords)
            raise
        return super(ast_expr_op, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op:
                arg0 = ast_expr_op(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op("""%s""")' % s
        else:
            return 'isl.ast_expr_op("%s")' % s
    def arg(arg0, arg1):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_expr_op_get_arg(arg0.ptr, arg1)
        obj = ast_expr(ctx=ctx, ptr=res)
        return obj
    def get_arg(arg0, arg1):
        return arg0.arg(arg1)
    def n_arg(arg0):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_expr_op_get_n_arg(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def get_n_arg(arg0):
        return arg0.n_arg()

isl.isl_ast_expr_op_get_arg.restype = c_void_p
isl.isl_ast_expr_op_get_arg.argtypes = [c_void_p, c_int]
isl.isl_ast_expr_op_get_n_arg.argtypes = [c_void_p]
isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]
isl.isl_ast_expr_op_get_type.argtypes = [c_void_p]

class ast_expr_op_access(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_access, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_access:
                arg0 = ast_expr_op_access(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_access("""%s""")' % s
        else:
            return 'isl.ast_expr_op_access("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_add(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_add, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_add:
                arg0 = ast_expr_op_add(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_add("""%s""")' % s
        else:
            return 'isl.ast_expr_op_add("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_address_of(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_address_of, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_address_of:
                arg0 = ast_expr_op_address_of(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_address_of("""%s""")' % s
        else:
            return 'isl.ast_expr_op_address_of("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_and(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_and, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_and:
                arg0 = ast_expr_op_and(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_and("""%s""")' % s
        else:
            return 'isl.ast_expr_op_and("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_and_then(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_and_then, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_and_then:
                arg0 = ast_expr_op_and_then(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_and_then("""%s""")' % s
        else:
            return 'isl.ast_expr_op_and_then("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_call(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_call, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_call:
                arg0 = ast_expr_op_call(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_call("""%s""")' % s
        else:
            return 'isl.ast_expr_op_call("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_cond(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_cond, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_cond:
                arg0 = ast_expr_op_cond(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_cond("""%s""")' % s
        else:
            return 'isl.ast_expr_op_cond("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_div(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_div, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_div:
                arg0 = ast_expr_op_div(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_div("""%s""")' % s
        else:
            return 'isl.ast_expr_op_div("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_eq(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_eq, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_eq:
                arg0 = ast_expr_op_eq(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_eq("""%s""")' % s
        else:
            return 'isl.ast_expr_op_eq("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_fdiv_q(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_fdiv_q, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_fdiv_q:
                arg0 = ast_expr_op_fdiv_q(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_fdiv_q("""%s""")' % s
        else:
            return 'isl.ast_expr_op_fdiv_q("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_ge(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_ge, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_ge:
                arg0 = ast_expr_op_ge(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_ge("""%s""")' % s
        else:
            return 'isl.ast_expr_op_ge("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_gt(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_gt, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_gt:
                arg0 = ast_expr_op_gt(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_gt("""%s""")' % s
        else:
            return 'isl.ast_expr_op_gt("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_le(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_le, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_le:
                arg0 = ast_expr_op_le(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_le("""%s""")' % s
        else:
            return 'isl.ast_expr_op_le("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_lt(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_lt, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_lt:
                arg0 = ast_expr_op_lt(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_lt("""%s""")' % s
        else:
            return 'isl.ast_expr_op_lt("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_max(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_max, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_max:
                arg0 = ast_expr_op_max(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_max("""%s""")' % s
        else:
            return 'isl.ast_expr_op_max("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_member(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_member, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_member:
                arg0 = ast_expr_op_member(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_member("""%s""")' % s
        else:
            return 'isl.ast_expr_op_member("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_min(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_min, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_min:
                arg0 = ast_expr_op_min(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_min("""%s""")' % s
        else:
            return 'isl.ast_expr_op_min("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_minus(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_minus, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_minus:
                arg0 = ast_expr_op_minus(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_minus("""%s""")' % s
        else:
            return 'isl.ast_expr_op_minus("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_mul(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_mul, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_mul:
                arg0 = ast_expr_op_mul(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_mul("""%s""")' % s
        else:
            return 'isl.ast_expr_op_mul("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_or(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_or, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_or:
                arg0 = ast_expr_op_or(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_or("""%s""")' % s
        else:
            return 'isl.ast_expr_op_or("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_or_else(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_or_else, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_or_else:
                arg0 = ast_expr_op_or_else(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_or_else("""%s""")' % s
        else:
            return 'isl.ast_expr_op_or_else("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_pdiv_q(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_pdiv_q, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_pdiv_q:
                arg0 = ast_expr_op_pdiv_q(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_pdiv_q("""%s""")' % s
        else:
            return 'isl.ast_expr_op_pdiv_q("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_pdiv_r(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_pdiv_r, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_pdiv_r:
                arg0 = ast_expr_op_pdiv_r(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_pdiv_r("""%s""")' % s
        else:
            return 'isl.ast_expr_op_pdiv_r("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_select(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_select, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_select:
                arg0 = ast_expr_op_select(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_select("""%s""")' % s
        else:
            return 'isl.ast_expr_op_select("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_sub(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_sub, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_sub:
                arg0 = ast_expr_op_sub(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_sub("""%s""")' % s
        else:
            return 'isl.ast_expr_op_sub("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_expr_op_zdiv_r(ast_expr_op):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_expr_op_zdiv_r, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr_op_zdiv_r:
                arg0 = ast_expr_op_zdiv_r(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr_op_zdiv_r("""%s""")' % s
        else:
            return 'isl.ast_expr_op_zdiv_r("%s")' % s

isl.isl_ast_expr_copy.restype = c_void_p
isl.isl_ast_expr_copy.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_node(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and isinstance(args[0], ast_node_for):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_node_if):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_node_block):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_node_mark):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], ast_node_user):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_ast_node_copy(args[0].ptr)
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        if "ptr" in keywords:
            type = isl.isl_ast_node_get_type(keywords["ptr"])
            if type == 1:
                return ast_node_for(**keywords)
            if type == 2:
                return ast_node_if(**keywords)
            if type == 3:
                return ast_node_block(**keywords)
            if type == 4:
                return ast_node_mark(**keywords)
            if type == 5:
                return ast_node_user(**keywords)
            raise
        return super(ast_node, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node("""%s""")' % s
        else:
            return 'isl.ast_node("%s")' % s
    def to_C_str(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_to_C_str(arg0.ptr)
        if res == 0:
            raise
        string = cast(res, c_char_p).value.decode('ascii')
        libc.free(res)
        return string

isl.isl_ast_node_to_C_str.restype = POINTER(c_char)
isl.isl_ast_node_to_C_str.argtypes = [c_void_p]
isl.isl_ast_node_copy.restype = c_void_p
isl.isl_ast_node_copy.argtypes = [c_void_p]
isl.isl_ast_node_free.restype = c_void_p
isl.isl_ast_node_free.argtypes = [c_void_p]
isl.isl_ast_node_to_str.restype = POINTER(c_char)
isl.isl_ast_node_to_str.argtypes = [c_void_p]
isl.isl_ast_node_get_type.argtypes = [c_void_p]

class ast_node_block(ast_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_node_block, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node_block:
                arg0 = ast_node_block(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node_block("""%s""")' % s
        else:
            return 'isl.ast_node_block("%s")' % s
    def children(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_block_get_children(arg0.ptr)
        obj = ast_node_list(ctx=ctx, ptr=res)
        return obj
    def get_children(arg0):
        return arg0.children()

isl.isl_ast_node_block_get_children.restype = c_void_p
isl.isl_ast_node_block_get_children.argtypes = [c_void_p]
isl.isl_ast_node_copy.restype = c_void_p
isl.isl_ast_node_copy.argtypes = [c_void_p]
isl.isl_ast_node_free.restype = c_void_p
isl.isl_ast_node_free.argtypes = [c_void_p]
isl.isl_ast_node_to_str.restype = POINTER(c_char)
isl.isl_ast_node_to_str.argtypes = [c_void_p]

class ast_node_for(ast_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_node_for, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node_for:
                arg0 = ast_node_for(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node_for("""%s""")' % s
        else:
            return 'isl.ast_node_for("%s")' % s
    def body(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_for_get_body(arg0.ptr)
        obj = ast_node(ctx=ctx, ptr=res)
        return obj
    def get_body(arg0):
        return arg0.body()
    def cond(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_for_get_cond(arg0.ptr)
        obj = ast_expr(ctx=ctx, ptr=res)
        return obj
    def get_cond(arg0):
        return arg0.cond()
    def inc(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_for_get_inc(arg0.ptr)
        obj = ast_expr(ctx=ctx, ptr=res)
        return obj
    def get_inc(arg0):
        return arg0.inc()
    def init(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_for_get_init(arg0.ptr)
        obj = ast_expr(ctx=ctx, ptr=res)
        return obj
    def get_init(arg0):
        return arg0.init()
    def iterator(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_for_get_iterator(arg0.ptr)
        obj = ast_expr(ctx=ctx, ptr=res)
        return obj
    def get_iterator(arg0):
        return arg0.iterator()
    def is_degenerate(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_for_is_degenerate(arg0.ptr)
        if res < 0:
            raise
        return bool(res)

isl.isl_ast_node_for_get_body.restype = c_void_p
isl.isl_ast_node_for_get_body.argtypes = [c_void_p]
isl.isl_ast_node_for_get_cond.restype = c_void_p
isl.isl_ast_node_for_get_cond.argtypes = [c_void_p]
isl.isl_ast_node_for_get_inc.restype = c_void_p
isl.isl_ast_node_for_get_inc.argtypes = [c_void_p]
isl.isl_ast_node_for_get_init.restype = c_void_p
isl.isl_ast_node_for_get_init.argtypes = [c_void_p]
isl.isl_ast_node_for_get_iterator.restype = c_void_p
isl.isl_ast_node_for_get_iterator.argtypes = [c_void_p]
isl.isl_ast_node_for_is_degenerate.argtypes = [c_void_p]
isl.isl_ast_node_copy.restype = c_void_p
isl.isl_ast_node_copy.argtypes = [c_void_p]
isl.isl_ast_node_free.restype = c_void_p
isl.isl_ast_node_free.argtypes = [c_void_p]
isl.isl_ast_node_to_str.restype = POINTER(c_char)
isl.isl_ast_node_to_str.argtypes = [c_void_p]

class ast_node_if(ast_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_node_if, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node_if:
                arg0 = ast_node_if(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node_if("""%s""")' % s
        else:
            return 'isl.ast_node_if("%s")' % s
    def cond(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_if_get_cond(arg0.ptr)
        obj = ast_expr(ctx=ctx, ptr=res)
        return obj
    def get_cond(arg0):
        return arg0.cond()
    def else_node(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_if_get_else_node(arg0.ptr)
        obj = ast_node(ctx=ctx, ptr=res)
        return obj
    def get_else_node(arg0):
        return arg0.else_node()
    def then_node(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_if_get_then_node(arg0.ptr)
        obj = ast_node(ctx=ctx, ptr=res)
        return obj
    def get_then_node(arg0):
        return arg0.then_node()
    def has_else_node(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_if_has_else_node(arg0.ptr)
        if res < 0:
            raise
        return bool(res)

isl.isl_ast_node_if_get_cond.restype = c_void_p
isl.isl_ast_node_if_get_cond.argtypes = [c_void_p]
isl.isl_ast_node_if_get_else_node.restype = c_void_p
isl.isl_ast_node_if_get_else_node.argtypes = [c_void_p]
isl.isl_ast_node_if_get_then_node.restype = c_void_p
isl.isl_ast_node_if_get_then_node.argtypes = [c_void_p]
isl.isl_ast_node_if_has_else_node.argtypes = [c_void_p]
isl.isl_ast_node_copy.restype = c_void_p
isl.isl_ast_node_copy.argtypes = [c_void_p]
isl.isl_ast_node_free.restype = c_void_p
isl.isl_ast_node_free.argtypes = [c_void_p]
isl.isl_ast_node_to_str.restype = POINTER(c_char)
isl.isl_ast_node_to_str.argtypes = [c_void_p]

class ast_node_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_ast_node_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is ast_node:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_ast_node_list_from_ast_node(isl.isl_ast_node_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node_list("""%s""")' % s
        else:
            return 'isl.ast_node_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is ast_node:
                arg1 = ast_node(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_add(isl.isl_ast_node_list_copy(arg0.ptr), isl.isl_ast_node_copy(arg1.ptr))
        obj = ast_node_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_clear(isl.isl_ast_node_list_copy(arg0.ptr))
        obj = ast_node_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is ast_node_list:
                arg1 = ast_node_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_concat(isl.isl_ast_node_list_copy(arg0.ptr), isl.isl_ast_node_list_copy(arg1.ptr))
        obj = ast_node_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_drop(isl.isl_ast_node_list_copy(arg0.ptr), arg1, arg2)
        obj = ast_node_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = ast_node(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_get_at(arg0.ptr, arg1)
        obj = ast_node(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is ast_node:
                arg2 = ast_node(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_insert(isl.isl_ast_node_list_copy(arg0.ptr), arg1, isl.isl_ast_node_copy(arg2.ptr))
        obj = ast_node_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is ast_node_list:
                arg0 = ast_node_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_ast_node_list_alloc.restype = c_void_p
isl.isl_ast_node_list_alloc.argtypes = [Context, c_int]
isl.isl_ast_node_list_from_ast_node.restype = c_void_p
isl.isl_ast_node_list_from_ast_node.argtypes = [c_void_p]
isl.isl_ast_node_list_add.restype = c_void_p
isl.isl_ast_node_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_ast_node_list_clear.restype = c_void_p
isl.isl_ast_node_list_clear.argtypes = [c_void_p]
isl.isl_ast_node_list_concat.restype = c_void_p
isl.isl_ast_node_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_ast_node_list_drop.restype = c_void_p
isl.isl_ast_node_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_ast_node_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_ast_node_list_get_at.restype = c_void_p
isl.isl_ast_node_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_ast_node_list_insert.restype = c_void_p
isl.isl_ast_node_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_ast_node_list_size.argtypes = [c_void_p]
isl.isl_ast_node_list_copy.restype = c_void_p
isl.isl_ast_node_list_copy.argtypes = [c_void_p]
isl.isl_ast_node_list_free.restype = c_void_p
isl.isl_ast_node_list_free.argtypes = [c_void_p]
isl.isl_ast_node_list_to_str.restype = POINTER(c_char)
isl.isl_ast_node_list_to_str.argtypes = [c_void_p]

class ast_node_mark(ast_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_node_mark, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node_mark:
                arg0 = ast_node_mark(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node_mark("""%s""")' % s
        else:
            return 'isl.ast_node_mark("%s")' % s
    def id(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_mark_get_id(arg0.ptr)
        obj = id(ctx=ctx, ptr=res)
        return obj
    def get_id(arg0):
        return arg0.id()
    def node(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_mark_get_node(arg0.ptr)
        obj = ast_node(ctx=ctx, ptr=res)
        return obj
    def get_node(arg0):
        return arg0.node()

isl.isl_ast_node_mark_get_id.restype = c_void_p
isl.isl_ast_node_mark_get_id.argtypes = [c_void_p]
isl.isl_ast_node_mark_get_node.restype = c_void_p
isl.isl_ast_node_mark_get_node.argtypes = [c_void_p]
isl.isl_ast_node_copy.restype = c_void_p
isl.isl_ast_node_copy.argtypes = [c_void_p]
isl.isl_ast_node_free.restype = c_void_p
isl.isl_ast_node_free.argtypes = [c_void_p]
isl.isl_ast_node_to_str.restype = POINTER(c_char)
isl.isl_ast_node_to_str.argtypes = [c_void_p]

class ast_node_user(ast_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(ast_node_user, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node_user:
                arg0 = ast_node_user(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node_user("""%s""")' % s
        else:
            return 'isl.ast_node_user("%s")' % s
    def expr(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_user_get_expr(arg0.ptr)
        obj = ast_expr(ctx=ctx, ptr=res)
        return obj
    def get_expr(arg0):
        return arg0.expr()

isl.isl_ast_node_user_get_expr.restype = c_void_p
isl.isl_ast_node_user_get_expr.argtypes = [c_void_p]
isl.isl_ast_node_copy.restype = c_void_p
isl.isl_ast_node_copy.argtypes = [c_void_p]
isl.isl_ast_node_free.restype = c_void_p
isl.isl_ast_node_free.argtypes = [c_void_p]
isl.isl_ast_node_to_str.restype = POINTER(c_char)
isl.isl_ast_node_to_str.argtypes = [c_void_p]

class union_map(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is basic_map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_map_from_basic_map(isl.isl_basic_map_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_map_from_map(isl.isl_map_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_map_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_map_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ptr = isl.isl_union_map_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_map("""%s""")' % s
        else:
            return 'isl.union_map("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_affine_hull(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def apply_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_apply_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def apply_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_apply_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def bind_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_bind_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_coalesce(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def compute_divs(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_compute_divs(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def curry(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_curry(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def deltas(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_deltas(isl.isl_union_map_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_detect_equalities(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain(isl.isl_union_map_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def domain_factor_domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_factor_domain(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def domain_factor_range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_factor_range(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def domain_map(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_map(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def domain_map_union_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_map_union_pw_multi_aff(isl.isl_union_map_copy(arg0.ptr))
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def domain_product(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_product(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def empty(*args):
        if len(args) == 0:
            ctx = Context.getDefaultInstance()
            res = isl.isl_union_map_empty_ctx(ctx)
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def eq_at(*args):
        if len(args) == 2 and args[1].__class__ is multi_union_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_eq_at_multi_union_pw_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_multi_union_pw_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def every_map(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = map(ctx=arg0.ctx, ptr=isl.isl_map_copy(cb_arg0))
            try:
                res = arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 1 if res else 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_map_every_map(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
        return bool(res)
    def extract_map(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_extract_map(arg0.ptr, isl.isl_space_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def factor_domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_factor_domain(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def factor_range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_factor_range(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def fixed_power(*args):
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_union_map_fixed_power_val(isl.isl_union_map_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def foreach_map(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = map(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_map_foreach_map(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    @staticmethod
    def convert_from(*args):
        if len(args) == 1 and args[0].__class__ is multi_union_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_from_multi_union_pw_aff(isl.isl_multi_union_pw_aff_copy(args[0].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 1 and args[0].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_from_union_pw_multi_aff(isl.isl_union_pw_multi_aff_copy(args[0].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    @staticmethod
    def from_domain(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_from_domain(isl.isl_union_set_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def from_domain_and_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_from_domain_and_range(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def from_range(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_from_range(isl.isl_union_set_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def space(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def gist_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def gist_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist_params(isl.isl_union_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def gist_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(*args):
        if len(args) == 2 and args[1].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_union_map_intersect_domain_space(isl.isl_union_map_copy(args[0].ptr), isl.isl_space_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_set:
            ctx = args[0].ctx
            res = isl.isl_union_map_intersect_domain_union_set(isl.isl_union_map_copy(args[0].ptr), isl.isl_union_set_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def intersect_domain_factor_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_domain_factor_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def intersect_domain_factor_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_domain_factor_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_params(isl.isl_union_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def intersect_range(*args):
        if len(args) == 2 and args[1].__class__ is space:
            ctx = args[0].ctx
            res = isl.isl_union_map_intersect_range_space(isl.isl_union_map_copy(args[0].ptr), isl.isl_space_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_set:
            ctx = args[0].ctx
            res = isl.isl_union_map_intersect_range_union_set(isl.isl_union_map_copy(args[0].ptr), isl.isl_union_set_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def intersect_range_factor_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_range_factor_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def intersect_range_factor_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_range_factor_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def is_bijective(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_bijective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_disjoint(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_disjoint(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_injective(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_injective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_single_valued(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_single_valued(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def isa_map(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_isa_map(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_lexmax(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def lexmin(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_lexmin(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_polyhedral_hull(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def preimage_domain(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_preimage_domain_multi_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_preimage_domain_multi_pw_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_preimage_domain_pw_multi_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_preimage_domain_union_pw_multi_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def preimage_range(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_preimage_range_multi_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_preimage_range_pw_multi_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_map_preimage_range_union_pw_multi_aff(isl.isl_union_map_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = union_map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_product(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def project_out_all_params(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_project_out_all_params(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range(isl.isl_union_map_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def range_factor_domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_factor_domain(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def range_factor_range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_factor_range(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def range_map(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_map(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_product(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def range_reverse(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_reverse(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def reverse(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_reverse(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_subtract(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def subtract_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_subtract_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def subtract_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_subtract_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def uncurry(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_uncurry(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_union(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def universe(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_universe(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def wrap(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_wrap(isl.isl_union_map_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def zip(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_zip(isl.isl_union_map_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj

isl.isl_union_map_from_basic_map.restype = c_void_p
isl.isl_union_map_from_basic_map.argtypes = [c_void_p]
isl.isl_union_map_from_map.restype = c_void_p
isl.isl_union_map_from_map.argtypes = [c_void_p]
isl.isl_union_map_read_from_str.restype = c_void_p
isl.isl_union_map_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_map_affine_hull.restype = c_void_p
isl.isl_union_map_affine_hull.argtypes = [c_void_p]
isl.isl_union_map_apply_domain.restype = c_void_p
isl.isl_union_map_apply_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_apply_range.restype = c_void_p
isl.isl_union_map_apply_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_bind_range.restype = c_void_p
isl.isl_union_map_bind_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_coalesce.restype = c_void_p
isl.isl_union_map_coalesce.argtypes = [c_void_p]
isl.isl_union_map_compute_divs.restype = c_void_p
isl.isl_union_map_compute_divs.argtypes = [c_void_p]
isl.isl_union_map_curry.restype = c_void_p
isl.isl_union_map_curry.argtypes = [c_void_p]
isl.isl_union_map_deltas.restype = c_void_p
isl.isl_union_map_deltas.argtypes = [c_void_p]
isl.isl_union_map_detect_equalities.restype = c_void_p
isl.isl_union_map_detect_equalities.argtypes = [c_void_p]
isl.isl_union_map_domain.restype = c_void_p
isl.isl_union_map_domain.argtypes = [c_void_p]
isl.isl_union_map_domain_factor_domain.restype = c_void_p
isl.isl_union_map_domain_factor_domain.argtypes = [c_void_p]
isl.isl_union_map_domain_factor_range.restype = c_void_p
isl.isl_union_map_domain_factor_range.argtypes = [c_void_p]
isl.isl_union_map_domain_map.restype = c_void_p
isl.isl_union_map_domain_map.argtypes = [c_void_p]
isl.isl_union_map_domain_map_union_pw_multi_aff.restype = c_void_p
isl.isl_union_map_domain_map_union_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_map_domain_product.restype = c_void_p
isl.isl_union_map_domain_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_empty_ctx.restype = c_void_p
isl.isl_union_map_empty_ctx.argtypes = [Context]
isl.isl_union_map_eq_at_multi_union_pw_aff.restype = c_void_p
isl.isl_union_map_eq_at_multi_union_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_every_map.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_map_extract_map.restype = c_void_p
isl.isl_union_map_extract_map.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_factor_domain.restype = c_void_p
isl.isl_union_map_factor_domain.argtypes = [c_void_p]
isl.isl_union_map_factor_range.restype = c_void_p
isl.isl_union_map_factor_range.argtypes = [c_void_p]
isl.isl_union_map_fixed_power_val.restype = c_void_p
isl.isl_union_map_fixed_power_val.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_foreach_map.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_map_from_multi_union_pw_aff.restype = c_void_p
isl.isl_union_map_from_multi_union_pw_aff.argtypes = [c_void_p]
isl.isl_union_map_from_union_pw_multi_aff.restype = c_void_p
isl.isl_union_map_from_union_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_map_from_domain.restype = c_void_p
isl.isl_union_map_from_domain.argtypes = [c_void_p]
isl.isl_union_map_from_domain_and_range.restype = c_void_p
isl.isl_union_map_from_domain_and_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_from_range.restype = c_void_p
isl.isl_union_map_from_range.argtypes = [c_void_p]
isl.isl_union_map_get_space.restype = c_void_p
isl.isl_union_map_get_space.argtypes = [c_void_p]
isl.isl_union_map_gist.restype = c_void_p
isl.isl_union_map_gist.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_gist_domain.restype = c_void_p
isl.isl_union_map_gist_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_gist_params.restype = c_void_p
isl.isl_union_map_gist_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_gist_range.restype = c_void_p
isl.isl_union_map_gist_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect.restype = c_void_p
isl.isl_union_map_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_domain_space.restype = c_void_p
isl.isl_union_map_intersect_domain_space.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_domain_union_set.restype = c_void_p
isl.isl_union_map_intersect_domain_union_set.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_domain_factor_domain.restype = c_void_p
isl.isl_union_map_intersect_domain_factor_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_domain_factor_range.restype = c_void_p
isl.isl_union_map_intersect_domain_factor_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_params.restype = c_void_p
isl.isl_union_map_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_range_space.restype = c_void_p
isl.isl_union_map_intersect_range_space.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_range_union_set.restype = c_void_p
isl.isl_union_map_intersect_range_union_set.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_range_factor_domain.restype = c_void_p
isl.isl_union_map_intersect_range_factor_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_range_factor_range.restype = c_void_p
isl.isl_union_map_intersect_range_factor_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_is_bijective.argtypes = [c_void_p]
isl.isl_union_map_is_disjoint.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_is_empty.argtypes = [c_void_p]
isl.isl_union_map_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_is_injective.argtypes = [c_void_p]
isl.isl_union_map_is_single_valued.argtypes = [c_void_p]
isl.isl_union_map_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_isa_map.argtypes = [c_void_p]
isl.isl_union_map_lexmax.restype = c_void_p
isl.isl_union_map_lexmax.argtypes = [c_void_p]
isl.isl_union_map_lexmin.restype = c_void_p
isl.isl_union_map_lexmin.argtypes = [c_void_p]
isl.isl_union_map_polyhedral_hull.restype = c_void_p
isl.isl_union_map_polyhedral_hull.argtypes = [c_void_p]
isl.isl_union_map_preimage_domain_multi_aff.restype = c_void_p
isl.isl_union_map_preimage_domain_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_preimage_domain_multi_pw_aff.restype = c_void_p
isl.isl_union_map_preimage_domain_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_preimage_domain_pw_multi_aff.restype = c_void_p
isl.isl_union_map_preimage_domain_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_preimage_domain_union_pw_multi_aff.restype = c_void_p
isl.isl_union_map_preimage_domain_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_preimage_range_multi_aff.restype = c_void_p
isl.isl_union_map_preimage_range_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_preimage_range_pw_multi_aff.restype = c_void_p
isl.isl_union_map_preimage_range_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_preimage_range_union_pw_multi_aff.restype = c_void_p
isl.isl_union_map_preimage_range_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_product.restype = c_void_p
isl.isl_union_map_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_project_out_all_params.restype = c_void_p
isl.isl_union_map_project_out_all_params.argtypes = [c_void_p]
isl.isl_union_map_range.restype = c_void_p
isl.isl_union_map_range.argtypes = [c_void_p]
isl.isl_union_map_range_factor_domain.restype = c_void_p
isl.isl_union_map_range_factor_domain.argtypes = [c_void_p]
isl.isl_union_map_range_factor_range.restype = c_void_p
isl.isl_union_map_range_factor_range.argtypes = [c_void_p]
isl.isl_union_map_range_map.restype = c_void_p
isl.isl_union_map_range_map.argtypes = [c_void_p]
isl.isl_union_map_range_product.restype = c_void_p
isl.isl_union_map_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_range_reverse.restype = c_void_p
isl.isl_union_map_range_reverse.argtypes = [c_void_p]
isl.isl_union_map_reverse.restype = c_void_p
isl.isl_union_map_reverse.argtypes = [c_void_p]
isl.isl_union_map_subtract.restype = c_void_p
isl.isl_union_map_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_subtract_domain.restype = c_void_p
isl.isl_union_map_subtract_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_subtract_range.restype = c_void_p
isl.isl_union_map_subtract_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_uncurry.restype = c_void_p
isl.isl_union_map_uncurry.argtypes = [c_void_p]
isl.isl_union_map_union.restype = c_void_p
isl.isl_union_map_union.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_universe.restype = c_void_p
isl.isl_union_map_universe.argtypes = [c_void_p]
isl.isl_union_map_wrap.restype = c_void_p
isl.isl_union_map_wrap.argtypes = [c_void_p]
isl.isl_union_map_zip.restype = c_void_p
isl.isl_union_map_zip.argtypes = [c_void_p]
isl.isl_union_map_copy.restype = c_void_p
isl.isl_union_map_copy.argtypes = [c_void_p]
isl.isl_union_map_free.restype = c_void_p
isl.isl_union_map_free.argtypes = [c_void_p]
isl.isl_union_map_to_str.restype = POINTER(c_char)
isl.isl_union_map_to_str.argtypes = [c_void_p]

class map(union_map):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is basic_map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_map_from_basic_map(isl.isl_basic_map_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_map_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_map_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ptr = isl.isl_map_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.map("""%s""")' % s
        else:
            return 'isl.map("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_affine_hull(isl.isl_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def apply_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).apply_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_apply_domain(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def apply_range(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).apply_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_apply_range(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def bind_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_map(arg0).bind_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_bind_domain(isl.isl_map_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def bind_range(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_map(arg0).bind_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_bind_range(isl.isl_map_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_coalesce(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def complement(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_complement(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def curry(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_curry(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def deltas(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_deltas(isl.isl_map_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_detect_equalities(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_domain(isl.isl_map_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def domain_factor_domain(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_domain_factor_domain(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def domain_factor_range(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_domain_factor_range(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def domain_product(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).domain_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_domain_product(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def empty(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_empty(isl.isl_space_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def eq_at(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_eq_at_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def factor_domain(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_factor_domain(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def factor_range(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_factor_range(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def flatten(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_flatten(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def flatten_domain(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_flatten_domain(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def flatten_range(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_flatten_range(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def foreach_basic_map(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = basic_map(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_map_foreach_basic_map(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def range_simple_fixed_box_hull(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_get_range_simple_fixed_box_hull(arg0.ptr)
        obj = fixed_box(ctx=ctx, ptr=res)
        return obj
    def get_range_simple_fixed_box_hull(arg0):
        return arg0.range_simple_fixed_box_hull()
    def space(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_gist(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def gist_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).gist_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_gist_domain(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).intersect_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_domain(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect_domain_factor_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).intersect_domain_factor_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_domain_factor_domain(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect_domain_factor_range(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).intersect_domain_factor_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_domain_factor_range(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_params(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect_range(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).intersect_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_range(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect_range_factor_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).intersect_range_factor_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_range_factor_domain(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect_range_factor_range(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).intersect_range_factor_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_range_factor_range(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def is_bijective(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_bijective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_disjoint(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_disjoint(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_disjoint(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_injective(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_injective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_single_valued(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_single_valued(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_strict_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lex_ge_at(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_lex_ge_at_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def lex_gt_at(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_lex_gt_at_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def lex_le_at(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_lex_le_at_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def lex_lt_at(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_lex_lt_at_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def lexmax(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_lexmax(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def lexmax_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_lexmax_pw_multi_aff(isl.isl_map_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def lexmin(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_lexmin(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def lexmin_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_lexmin_pw_multi_aff(isl.isl_map_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def lower_bound(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_lower_bound_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def max_multi_pw_aff(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_max_multi_pw_aff(isl.isl_map_copy(arg0.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def min_multi_pw_aff(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_min_multi_pw_aff(isl.isl_map_copy(arg0.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_polyhedral_hull(isl.isl_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def preimage_domain(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_map_preimage_domain_multi_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_preimage_domain_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_map_preimage_domain_pw_multi_aff(isl.isl_map_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def preimage_range(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_map_preimage_range_multi_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_map_preimage_range_pw_multi_aff(isl.isl_map_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_product(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def project_out_all_params(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_project_out_all_params(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def range(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_range(isl.isl_map_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def range_factor_domain(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_range_factor_domain(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def range_factor_range(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_range_factor_range(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_range_product(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def range_reverse(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_range_reverse(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def reverse(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_reverse(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def sample(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_sample(isl.isl_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).subtract(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_subtract(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def uncurry(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_uncurry(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_union(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def universe(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_universe(isl.isl_space_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def unshifted_simple_hull(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_unshifted_simple_hull(isl.isl_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def upper_bound(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_map_upper_bound_multi_pw_aff(isl.isl_map_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = map(ctx=ctx, ptr=res)
            return obj
        raise Error
    def wrap(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_wrap(isl.isl_map_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def zip(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_zip(isl.isl_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj

isl.isl_map_from_basic_map.restype = c_void_p
isl.isl_map_from_basic_map.argtypes = [c_void_p]
isl.isl_map_read_from_str.restype = c_void_p
isl.isl_map_read_from_str.argtypes = [Context, c_char_p]
isl.isl_map_affine_hull.restype = c_void_p
isl.isl_map_affine_hull.argtypes = [c_void_p]
isl.isl_map_apply_domain.restype = c_void_p
isl.isl_map_apply_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_apply_range.restype = c_void_p
isl.isl_map_apply_range.argtypes = [c_void_p, c_void_p]
isl.isl_map_bind_domain.restype = c_void_p
isl.isl_map_bind_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_bind_range.restype = c_void_p
isl.isl_map_bind_range.argtypes = [c_void_p, c_void_p]
isl.isl_map_coalesce.restype = c_void_p
isl.isl_map_coalesce.argtypes = [c_void_p]
isl.isl_map_complement.restype = c_void_p
isl.isl_map_complement.argtypes = [c_void_p]
isl.isl_map_curry.restype = c_void_p
isl.isl_map_curry.argtypes = [c_void_p]
isl.isl_map_deltas.restype = c_void_p
isl.isl_map_deltas.argtypes = [c_void_p]
isl.isl_map_detect_equalities.restype = c_void_p
isl.isl_map_detect_equalities.argtypes = [c_void_p]
isl.isl_map_domain.restype = c_void_p
isl.isl_map_domain.argtypes = [c_void_p]
isl.isl_map_domain_factor_domain.restype = c_void_p
isl.isl_map_domain_factor_domain.argtypes = [c_void_p]
isl.isl_map_domain_factor_range.restype = c_void_p
isl.isl_map_domain_factor_range.argtypes = [c_void_p]
isl.isl_map_domain_product.restype = c_void_p
isl.isl_map_domain_product.argtypes = [c_void_p, c_void_p]
isl.isl_map_empty.restype = c_void_p
isl.isl_map_empty.argtypes = [c_void_p]
isl.isl_map_eq_at_multi_pw_aff.restype = c_void_p
isl.isl_map_eq_at_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_factor_domain.restype = c_void_p
isl.isl_map_factor_domain.argtypes = [c_void_p]
isl.isl_map_factor_range.restype = c_void_p
isl.isl_map_factor_range.argtypes = [c_void_p]
isl.isl_map_flatten.restype = c_void_p
isl.isl_map_flatten.argtypes = [c_void_p]
isl.isl_map_flatten_domain.restype = c_void_p
isl.isl_map_flatten_domain.argtypes = [c_void_p]
isl.isl_map_flatten_range.restype = c_void_p
isl.isl_map_flatten_range.argtypes = [c_void_p]
isl.isl_map_foreach_basic_map.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_map_get_range_simple_fixed_box_hull.restype = c_void_p
isl.isl_map_get_range_simple_fixed_box_hull.argtypes = [c_void_p]
isl.isl_map_get_space.restype = c_void_p
isl.isl_map_get_space.argtypes = [c_void_p]
isl.isl_map_gist.restype = c_void_p
isl.isl_map_gist.argtypes = [c_void_p, c_void_p]
isl.isl_map_gist_domain.restype = c_void_p
isl.isl_map_gist_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect.restype = c_void_p
isl.isl_map_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_domain.restype = c_void_p
isl.isl_map_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_domain_factor_domain.restype = c_void_p
isl.isl_map_intersect_domain_factor_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_domain_factor_range.restype = c_void_p
isl.isl_map_intersect_domain_factor_range.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_params.restype = c_void_p
isl.isl_map_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_range.restype = c_void_p
isl.isl_map_intersect_range.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_range_factor_domain.restype = c_void_p
isl.isl_map_intersect_range_factor_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_range_factor_range.restype = c_void_p
isl.isl_map_intersect_range_factor_range.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_bijective.argtypes = [c_void_p]
isl.isl_map_is_disjoint.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_empty.argtypes = [c_void_p]
isl.isl_map_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_injective.argtypes = [c_void_p]
isl.isl_map_is_single_valued.argtypes = [c_void_p]
isl.isl_map_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_map_lex_ge_at_multi_pw_aff.restype = c_void_p
isl.isl_map_lex_ge_at_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_lex_gt_at_multi_pw_aff.restype = c_void_p
isl.isl_map_lex_gt_at_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_lex_le_at_multi_pw_aff.restype = c_void_p
isl.isl_map_lex_le_at_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_lex_lt_at_multi_pw_aff.restype = c_void_p
isl.isl_map_lex_lt_at_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_lexmax.restype = c_void_p
isl.isl_map_lexmax.argtypes = [c_void_p]
isl.isl_map_lexmax_pw_multi_aff.restype = c_void_p
isl.isl_map_lexmax_pw_multi_aff.argtypes = [c_void_p]
isl.isl_map_lexmin.restype = c_void_p
isl.isl_map_lexmin.argtypes = [c_void_p]
isl.isl_map_lexmin_pw_multi_aff.restype = c_void_p
isl.isl_map_lexmin_pw_multi_aff.argtypes = [c_void_p]
isl.isl_map_lower_bound_multi_pw_aff.restype = c_void_p
isl.isl_map_lower_bound_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_max_multi_pw_aff.restype = c_void_p
isl.isl_map_max_multi_pw_aff.argtypes = [c_void_p]
isl.isl_map_min_multi_pw_aff.restype = c_void_p
isl.isl_map_min_multi_pw_aff.argtypes = [c_void_p]
isl.isl_map_polyhedral_hull.restype = c_void_p
isl.isl_map_polyhedral_hull.argtypes = [c_void_p]
isl.isl_map_preimage_domain_multi_aff.restype = c_void_p
isl.isl_map_preimage_domain_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_preimage_domain_multi_pw_aff.restype = c_void_p
isl.isl_map_preimage_domain_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_preimage_domain_pw_multi_aff.restype = c_void_p
isl.isl_map_preimage_domain_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_preimage_range_multi_aff.restype = c_void_p
isl.isl_map_preimage_range_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_preimage_range_pw_multi_aff.restype = c_void_p
isl.isl_map_preimage_range_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_product.restype = c_void_p
isl.isl_map_product.argtypes = [c_void_p, c_void_p]
isl.isl_map_project_out_all_params.restype = c_void_p
isl.isl_map_project_out_all_params.argtypes = [c_void_p]
isl.isl_map_range.restype = c_void_p
isl.isl_map_range.argtypes = [c_void_p]
isl.isl_map_range_factor_domain.restype = c_void_p
isl.isl_map_range_factor_domain.argtypes = [c_void_p]
isl.isl_map_range_factor_range.restype = c_void_p
isl.isl_map_range_factor_range.argtypes = [c_void_p]
isl.isl_map_range_product.restype = c_void_p
isl.isl_map_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_map_range_reverse.restype = c_void_p
isl.isl_map_range_reverse.argtypes = [c_void_p]
isl.isl_map_reverse.restype = c_void_p
isl.isl_map_reverse.argtypes = [c_void_p]
isl.isl_map_sample.restype = c_void_p
isl.isl_map_sample.argtypes = [c_void_p]
isl.isl_map_subtract.restype = c_void_p
isl.isl_map_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_map_uncurry.restype = c_void_p
isl.isl_map_uncurry.argtypes = [c_void_p]
isl.isl_map_union.restype = c_void_p
isl.isl_map_union.argtypes = [c_void_p, c_void_p]
isl.isl_map_universe.restype = c_void_p
isl.isl_map_universe.argtypes = [c_void_p]
isl.isl_map_unshifted_simple_hull.restype = c_void_p
isl.isl_map_unshifted_simple_hull.argtypes = [c_void_p]
isl.isl_map_upper_bound_multi_pw_aff.restype = c_void_p
isl.isl_map_upper_bound_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_map_wrap.restype = c_void_p
isl.isl_map_wrap.argtypes = [c_void_p]
isl.isl_map_zip.restype = c_void_p
isl.isl_map_zip.argtypes = [c_void_p]
isl.isl_map_copy.restype = c_void_p
isl.isl_map_copy.argtypes = [c_void_p]
isl.isl_map_free.restype = c_void_p
isl.isl_map_free.argtypes = [c_void_p]
isl.isl_map_to_str.restype = POINTER(c_char)
isl.isl_map_to_str.argtypes = [c_void_p]

class basic_map(map):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_basic_map_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_basic_map_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ptr = isl.isl_basic_map_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.basic_map("""%s""")' % s
        else:
            return 'isl.basic_map("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_affine_hull(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def apply_domain(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).apply_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_apply_domain(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def apply_range(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).apply_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_apply_range(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def deltas(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_deltas(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_detect_equalities(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def flatten(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_flatten(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def flatten_domain(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_flatten_domain(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def flatten_range(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_flatten_range(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_gist(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_intersect(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return map(arg0).intersect_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_intersect_domain(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def intersect_range(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return map(arg0).intersect_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_intersect_range(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def is_empty(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_lexmax(isl.isl_basic_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def lexmin(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_lexmin(isl.isl_basic_map_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def reverse(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_reverse(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def sample(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_sample(isl.isl_basic_map_copy(arg0.ptr))
        obj = basic_map(ctx=ctx, ptr=res)
        return obj
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_union(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj

isl.isl_basic_map_read_from_str.restype = c_void_p
isl.isl_basic_map_read_from_str.argtypes = [Context, c_char_p]
isl.isl_basic_map_affine_hull.restype = c_void_p
isl.isl_basic_map_affine_hull.argtypes = [c_void_p]
isl.isl_basic_map_apply_domain.restype = c_void_p
isl.isl_basic_map_apply_domain.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_apply_range.restype = c_void_p
isl.isl_basic_map_apply_range.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_deltas.restype = c_void_p
isl.isl_basic_map_deltas.argtypes = [c_void_p]
isl.isl_basic_map_detect_equalities.restype = c_void_p
isl.isl_basic_map_detect_equalities.argtypes = [c_void_p]
isl.isl_basic_map_flatten.restype = c_void_p
isl.isl_basic_map_flatten.argtypes = [c_void_p]
isl.isl_basic_map_flatten_domain.restype = c_void_p
isl.isl_basic_map_flatten_domain.argtypes = [c_void_p]
isl.isl_basic_map_flatten_range.restype = c_void_p
isl.isl_basic_map_flatten_range.argtypes = [c_void_p]
isl.isl_basic_map_gist.restype = c_void_p
isl.isl_basic_map_gist.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_intersect.restype = c_void_p
isl.isl_basic_map_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_intersect_domain.restype = c_void_p
isl.isl_basic_map_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_intersect_range.restype = c_void_p
isl.isl_basic_map_intersect_range.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_is_empty.argtypes = [c_void_p]
isl.isl_basic_map_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_lexmax.restype = c_void_p
isl.isl_basic_map_lexmax.argtypes = [c_void_p]
isl.isl_basic_map_lexmin.restype = c_void_p
isl.isl_basic_map_lexmin.argtypes = [c_void_p]
isl.isl_basic_map_reverse.restype = c_void_p
isl.isl_basic_map_reverse.argtypes = [c_void_p]
isl.isl_basic_map_sample.restype = c_void_p
isl.isl_basic_map_sample.argtypes = [c_void_p]
isl.isl_basic_map_union.restype = c_void_p
isl.isl_basic_map_union.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_copy.restype = c_void_p
isl.isl_basic_map_copy.argtypes = [c_void_p]
isl.isl_basic_map_free.restype = c_void_p
isl.isl_basic_map_free.argtypes = [c_void_p]
isl.isl_basic_map_to_str.restype = POINTER(c_char)
isl.isl_basic_map_to_str.argtypes = [c_void_p]

class union_set(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is basic_set:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_from_basic_set(isl.isl_basic_set_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is point:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_from_point(isl.isl_point_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is set:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_from_set(isl.isl_set_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_set_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ptr = isl.isl_union_set_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_set("""%s""")' % s
        else:
            return 'isl.union_set("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_affine_hull(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def apply(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_apply(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_coalesce(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def compute_divs(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_compute_divs(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_detect_equalities(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def empty(*args):
        if len(args) == 0:
            ctx = Context.getDefaultInstance()
            res = isl.isl_union_set_empty_ctx(ctx)
            obj = union_set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def every_set(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = set(ctx=arg0.ctx, ptr=isl.isl_set_copy(cb_arg0))
            try:
                res = arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 1 if res else 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_set_every_set(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
        return bool(res)
    def extract_set(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_extract_set(arg0.ptr, isl.isl_space_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def foreach_point(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = point(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_set_foreach_point(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def foreach_set(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = set(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_set_foreach_set(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def space(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_gist(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def gist_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_gist_params(isl.isl_union_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def identity(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_identity(isl.isl_union_set_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_intersect(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_intersect_params(isl.isl_union_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def is_disjoint(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_disjoint(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def isa_set(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_isa_set(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_lexmax(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def lexmin(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_lexmin(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_polyhedral_hull(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def preimage(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_set_preimage_multi_aff(isl.isl_union_set_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = union_set(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_set_preimage_pw_multi_aff(isl.isl_union_set_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = union_set(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_union_set_preimage_union_pw_multi_aff(isl.isl_union_set_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = union_set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def sample_point(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_sample_point(isl.isl_union_set_copy(arg0.ptr))
        obj = point(ctx=ctx, ptr=res)
        return obj
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_subtract(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_union(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def universe(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_universe(isl.isl_union_set_copy(arg0.ptr))
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def unwrap(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_unwrap(isl.isl_union_set_copy(arg0.ptr))
        obj = union_map(ctx=ctx, ptr=res)
        return obj

isl.isl_union_set_from_basic_set.restype = c_void_p
isl.isl_union_set_from_basic_set.argtypes = [c_void_p]
isl.isl_union_set_from_point.restype = c_void_p
isl.isl_union_set_from_point.argtypes = [c_void_p]
isl.isl_union_set_from_set.restype = c_void_p
isl.isl_union_set_from_set.argtypes = [c_void_p]
isl.isl_union_set_read_from_str.restype = c_void_p
isl.isl_union_set_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_set_affine_hull.restype = c_void_p
isl.isl_union_set_affine_hull.argtypes = [c_void_p]
isl.isl_union_set_apply.restype = c_void_p
isl.isl_union_set_apply.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_coalesce.restype = c_void_p
isl.isl_union_set_coalesce.argtypes = [c_void_p]
isl.isl_union_set_compute_divs.restype = c_void_p
isl.isl_union_set_compute_divs.argtypes = [c_void_p]
isl.isl_union_set_detect_equalities.restype = c_void_p
isl.isl_union_set_detect_equalities.argtypes = [c_void_p]
isl.isl_union_set_empty_ctx.restype = c_void_p
isl.isl_union_set_empty_ctx.argtypes = [Context]
isl.isl_union_set_every_set.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_set_extract_set.restype = c_void_p
isl.isl_union_set_extract_set.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_foreach_point.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_set_foreach_set.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_set_get_space.restype = c_void_p
isl.isl_union_set_get_space.argtypes = [c_void_p]
isl.isl_union_set_gist.restype = c_void_p
isl.isl_union_set_gist.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_gist_params.restype = c_void_p
isl.isl_union_set_gist_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_identity.restype = c_void_p
isl.isl_union_set_identity.argtypes = [c_void_p]
isl.isl_union_set_intersect.restype = c_void_p
isl.isl_union_set_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_intersect_params.restype = c_void_p
isl.isl_union_set_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_is_disjoint.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_is_empty.argtypes = [c_void_p]
isl.isl_union_set_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_isa_set.argtypes = [c_void_p]
isl.isl_union_set_lexmax.restype = c_void_p
isl.isl_union_set_lexmax.argtypes = [c_void_p]
isl.isl_union_set_lexmin.restype = c_void_p
isl.isl_union_set_lexmin.argtypes = [c_void_p]
isl.isl_union_set_polyhedral_hull.restype = c_void_p
isl.isl_union_set_polyhedral_hull.argtypes = [c_void_p]
isl.isl_union_set_preimage_multi_aff.restype = c_void_p
isl.isl_union_set_preimage_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_preimage_pw_multi_aff.restype = c_void_p
isl.isl_union_set_preimage_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_preimage_union_pw_multi_aff.restype = c_void_p
isl.isl_union_set_preimage_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_sample_point.restype = c_void_p
isl.isl_union_set_sample_point.argtypes = [c_void_p]
isl.isl_union_set_subtract.restype = c_void_p
isl.isl_union_set_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_union.restype = c_void_p
isl.isl_union_set_union.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_universe.restype = c_void_p
isl.isl_union_set_universe.argtypes = [c_void_p]
isl.isl_union_set_unwrap.restype = c_void_p
isl.isl_union_set_unwrap.argtypes = [c_void_p]
isl.isl_union_set_copy.restype = c_void_p
isl.isl_union_set_copy.argtypes = [c_void_p]
isl.isl_union_set_free.restype = c_void_p
isl.isl_union_set_free.argtypes = [c_void_p]
isl.isl_union_set_to_str.restype = POINTER(c_char)
isl.isl_union_set_to_str.argtypes = [c_void_p]

class set(union_set):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is basic_set:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_set_from_basic_set(isl.isl_basic_set_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is point:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_set_from_point(isl.isl_point_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_set_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_set_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ptr = isl.isl_set_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.set("""%s""")' % s
        else:
            return 'isl.set("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_affine_hull(isl.isl_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def apply(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_set(arg0).apply(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_apply(isl.isl_set_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def bind(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_set(arg0).bind(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_bind(isl.isl_set_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def coalesce(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_coalesce(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def complement(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_complement(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_detect_equalities(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def dim_max_val(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_dim_max_val(isl.isl_set_copy(arg0.ptr), arg1)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def dim_min_val(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_dim_min_val(isl.isl_set_copy(arg0.ptr), arg1)
        obj = val(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def empty(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_empty(isl.isl_space_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def flatten(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_flatten(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def foreach_basic_set(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = basic_set(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_set_foreach_basic_set(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def foreach_point(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = point(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_set_foreach_point(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def plain_multi_val_if_fixed(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_get_plain_multi_val_if_fixed(arg0.ptr)
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def get_plain_multi_val_if_fixed(arg0):
        return arg0.plain_multi_val_if_fixed()
    def simple_fixed_box_hull(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_get_simple_fixed_box_hull(arg0.ptr)
        obj = fixed_box(ctx=ctx, ptr=res)
        return obj
    def get_simple_fixed_box_hull(arg0):
        return arg0.simple_fixed_box_hull()
    def space(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def stride(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_get_stride(arg0.ptr, arg1)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def get_stride(arg0, arg1):
        return arg0.stride(arg1)
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_gist(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def identity(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_identity(isl.isl_set_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def indicator_function(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_indicator_function(isl.isl_set_copy(arg0.ptr))
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            return union_set(arg0).insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_insert_domain(isl.isl_set_copy(arg0.ptr), isl.isl_space_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_intersect(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_intersect_params(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def involves_locals(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_involves_locals(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_disjoint(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_disjoint(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_disjoint(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_singleton(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_is_singleton(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_strict_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_wrapping(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_is_wrapping(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_lexmax(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def lexmax_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_lexmax_pw_multi_aff(isl.isl_set_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def lexmin(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_lexmin(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def lexmin_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_lexmin_pw_multi_aff(isl.isl_set_copy(arg0.ptr))
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def lower_bound(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_set_lower_bound_multi_pw_aff(isl.isl_set_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_set_lower_bound_multi_val(isl.isl_set_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def max_multi_pw_aff(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_max_multi_pw_aff(isl.isl_set_copy(arg0.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def max_val(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return union_set(arg0).max_val(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_max_val(arg0.ptr, arg1.ptr)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def min_multi_pw_aff(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_min_multi_pw_aff(isl.isl_set_copy(arg0.ptr))
        obj = multi_pw_aff(ctx=ctx, ptr=res)
        return obj
    def min_val(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return union_set(arg0).min_val(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_min_val(arg0.ptr, arg1.ptr)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def params(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_params(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_polyhedral_hull(isl.isl_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def preimage(*args):
        if len(args) == 2 and args[1].__class__ is multi_aff:
            ctx = args[0].ctx
            res = isl.isl_set_preimage_multi_aff(isl.isl_set_copy(args[0].ptr), isl.isl_multi_aff_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_set_preimage_multi_pw_aff(isl.isl_set_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_set_preimage_pw_multi_aff(isl.isl_set_copy(args[0].ptr), isl.isl_pw_multi_aff_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_product(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def project_out_all_params(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_project_out_all_params(isl.isl_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def project_out_param(*args):
        if len(args) == 2 and (args[1].__class__ is id or type(args[1]) == str):
            args = list(args)
            try:
                if not args[1].__class__ is id:
                    args[1] = id(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_set_project_out_param_id(isl.isl_set_copy(args[0].ptr), isl.isl_id_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is id_list:
            ctx = args[0].ctx
            res = isl.isl_set_project_out_param_id_list(isl.isl_set_copy(args[0].ptr), isl.isl_id_list_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        raise Error
    def sample(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_sample(isl.isl_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def sample_point(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_sample_point(isl.isl_set_copy(arg0.ptr))
        obj = point(ctx=ctx, ptr=res)
        return obj
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).subtract(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_subtract(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def translation(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_translation(isl.isl_set_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def unbind_params(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_set(arg0).unbind_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_unbind_params(isl.isl_set_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def unbind_params_insert_domain(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            return union_set(arg0).unbind_params_insert_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_unbind_params_insert_domain(isl.isl_set_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_union(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def universe(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_universe(isl.isl_space_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def unshifted_simple_hull(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_unshifted_simple_hull(isl.isl_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def unwrap(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_unwrap(isl.isl_set_copy(arg0.ptr))
        obj = map(ctx=ctx, ptr=res)
        return obj
    def upper_bound(*args):
        if len(args) == 2 and args[1].__class__ is multi_pw_aff:
            ctx = args[0].ctx
            res = isl.isl_set_upper_bound_multi_pw_aff(isl.isl_set_copy(args[0].ptr), isl.isl_multi_pw_aff_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_set_upper_bound_multi_val(isl.isl_set_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = set(ctx=ctx, ptr=res)
            return obj
        raise Error

isl.isl_set_from_basic_set.restype = c_void_p
isl.isl_set_from_basic_set.argtypes = [c_void_p]
isl.isl_set_from_point.restype = c_void_p
isl.isl_set_from_point.argtypes = [c_void_p]
isl.isl_set_read_from_str.restype = c_void_p
isl.isl_set_read_from_str.argtypes = [Context, c_char_p]
isl.isl_set_affine_hull.restype = c_void_p
isl.isl_set_affine_hull.argtypes = [c_void_p]
isl.isl_set_apply.restype = c_void_p
isl.isl_set_apply.argtypes = [c_void_p, c_void_p]
isl.isl_set_bind.restype = c_void_p
isl.isl_set_bind.argtypes = [c_void_p, c_void_p]
isl.isl_set_coalesce.restype = c_void_p
isl.isl_set_coalesce.argtypes = [c_void_p]
isl.isl_set_complement.restype = c_void_p
isl.isl_set_complement.argtypes = [c_void_p]
isl.isl_set_detect_equalities.restype = c_void_p
isl.isl_set_detect_equalities.argtypes = [c_void_p]
isl.isl_set_dim_max_val.restype = c_void_p
isl.isl_set_dim_max_val.argtypes = [c_void_p, c_int]
isl.isl_set_dim_min_val.restype = c_void_p
isl.isl_set_dim_min_val.argtypes = [c_void_p, c_int]
isl.isl_set_empty.restype = c_void_p
isl.isl_set_empty.argtypes = [c_void_p]
isl.isl_set_flatten.restype = c_void_p
isl.isl_set_flatten.argtypes = [c_void_p]
isl.isl_set_foreach_basic_set.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_set_foreach_point.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_set_get_plain_multi_val_if_fixed.restype = c_void_p
isl.isl_set_get_plain_multi_val_if_fixed.argtypes = [c_void_p]
isl.isl_set_get_simple_fixed_box_hull.restype = c_void_p
isl.isl_set_get_simple_fixed_box_hull.argtypes = [c_void_p]
isl.isl_set_get_space.restype = c_void_p
isl.isl_set_get_space.argtypes = [c_void_p]
isl.isl_set_get_stride.restype = c_void_p
isl.isl_set_get_stride.argtypes = [c_void_p, c_int]
isl.isl_set_gist.restype = c_void_p
isl.isl_set_gist.argtypes = [c_void_p, c_void_p]
isl.isl_set_identity.restype = c_void_p
isl.isl_set_identity.argtypes = [c_void_p]
isl.isl_set_indicator_function.restype = c_void_p
isl.isl_set_indicator_function.argtypes = [c_void_p]
isl.isl_set_insert_domain.restype = c_void_p
isl.isl_set_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_set_intersect.restype = c_void_p
isl.isl_set_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_set_intersect_params.restype = c_void_p
isl.isl_set_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_set_involves_locals.argtypes = [c_void_p]
isl.isl_set_is_disjoint.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_empty.argtypes = [c_void_p]
isl.isl_set_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_singleton.argtypes = [c_void_p]
isl.isl_set_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_wrapping.argtypes = [c_void_p]
isl.isl_set_lexmax.restype = c_void_p
isl.isl_set_lexmax.argtypes = [c_void_p]
isl.isl_set_lexmax_pw_multi_aff.restype = c_void_p
isl.isl_set_lexmax_pw_multi_aff.argtypes = [c_void_p]
isl.isl_set_lexmin.restype = c_void_p
isl.isl_set_lexmin.argtypes = [c_void_p]
isl.isl_set_lexmin_pw_multi_aff.restype = c_void_p
isl.isl_set_lexmin_pw_multi_aff.argtypes = [c_void_p]
isl.isl_set_lower_bound_multi_pw_aff.restype = c_void_p
isl.isl_set_lower_bound_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_set_lower_bound_multi_val.restype = c_void_p
isl.isl_set_lower_bound_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_set_max_multi_pw_aff.restype = c_void_p
isl.isl_set_max_multi_pw_aff.argtypes = [c_void_p]
isl.isl_set_max_val.restype = c_void_p
isl.isl_set_max_val.argtypes = [c_void_p, c_void_p]
isl.isl_set_min_multi_pw_aff.restype = c_void_p
isl.isl_set_min_multi_pw_aff.argtypes = [c_void_p]
isl.isl_set_min_val.restype = c_void_p
isl.isl_set_min_val.argtypes = [c_void_p, c_void_p]
isl.isl_set_params.restype = c_void_p
isl.isl_set_params.argtypes = [c_void_p]
isl.isl_set_polyhedral_hull.restype = c_void_p
isl.isl_set_polyhedral_hull.argtypes = [c_void_p]
isl.isl_set_preimage_multi_aff.restype = c_void_p
isl.isl_set_preimage_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_set_preimage_multi_pw_aff.restype = c_void_p
isl.isl_set_preimage_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_set_preimage_pw_multi_aff.restype = c_void_p
isl.isl_set_preimage_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_set_product.restype = c_void_p
isl.isl_set_product.argtypes = [c_void_p, c_void_p]
isl.isl_set_project_out_all_params.restype = c_void_p
isl.isl_set_project_out_all_params.argtypes = [c_void_p]
isl.isl_set_project_out_param_id.restype = c_void_p
isl.isl_set_project_out_param_id.argtypes = [c_void_p, c_void_p]
isl.isl_set_project_out_param_id_list.restype = c_void_p
isl.isl_set_project_out_param_id_list.argtypes = [c_void_p, c_void_p]
isl.isl_set_sample.restype = c_void_p
isl.isl_set_sample.argtypes = [c_void_p]
isl.isl_set_sample_point.restype = c_void_p
isl.isl_set_sample_point.argtypes = [c_void_p]
isl.isl_set_subtract.restype = c_void_p
isl.isl_set_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_set_translation.restype = c_void_p
isl.isl_set_translation.argtypes = [c_void_p]
isl.isl_set_unbind_params.restype = c_void_p
isl.isl_set_unbind_params.argtypes = [c_void_p, c_void_p]
isl.isl_set_unbind_params_insert_domain.restype = c_void_p
isl.isl_set_unbind_params_insert_domain.argtypes = [c_void_p, c_void_p]
isl.isl_set_union.restype = c_void_p
isl.isl_set_union.argtypes = [c_void_p, c_void_p]
isl.isl_set_universe.restype = c_void_p
isl.isl_set_universe.argtypes = [c_void_p]
isl.isl_set_unshifted_simple_hull.restype = c_void_p
isl.isl_set_unshifted_simple_hull.argtypes = [c_void_p]
isl.isl_set_unwrap.restype = c_void_p
isl.isl_set_unwrap.argtypes = [c_void_p]
isl.isl_set_upper_bound_multi_pw_aff.restype = c_void_p
isl.isl_set_upper_bound_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_set_upper_bound_multi_val.restype = c_void_p
isl.isl_set_upper_bound_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_set_copy.restype = c_void_p
isl.isl_set_copy.argtypes = [c_void_p]
isl.isl_set_free.restype = c_void_p
isl.isl_set_free.argtypes = [c_void_p]
isl.isl_set_to_str.restype = POINTER(c_char)
isl.isl_set_to_str.argtypes = [c_void_p]

class basic_set(set):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is point:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_basic_set_from_point(isl.isl_point_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_basic_set_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_basic_set_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ptr = isl.isl_basic_set_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.basic_set("""%s""")' % s
        else:
            return 'isl.basic_set("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_affine_hull(isl.isl_basic_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def apply(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return set(arg0).apply(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_apply(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_detect_equalities(isl.isl_basic_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def dim_max_val(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_dim_max_val(isl.isl_basic_set_copy(arg0.ptr), arg1)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def flatten(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_flatten(isl.isl_basic_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_gist(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_intersect(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_intersect_params(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def is_empty(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_wrapping(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_wrapping(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_lexmax(isl.isl_basic_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def lexmin(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_lexmin(isl.isl_basic_set_copy(arg0.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj
    def params(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_params(isl.isl_basic_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def sample(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_sample(isl.isl_basic_set_copy(arg0.ptr))
        obj = basic_set(ctx=ctx, ptr=res)
        return obj
    def sample_point(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_sample_point(isl.isl_basic_set_copy(arg0.ptr))
        obj = point(ctx=ctx, ptr=res)
        return obj
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_union(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        obj = set(ctx=ctx, ptr=res)
        return obj

isl.isl_basic_set_from_point.restype = c_void_p
isl.isl_basic_set_from_point.argtypes = [c_void_p]
isl.isl_basic_set_read_from_str.restype = c_void_p
isl.isl_basic_set_read_from_str.argtypes = [Context, c_char_p]
isl.isl_basic_set_affine_hull.restype = c_void_p
isl.isl_basic_set_affine_hull.argtypes = [c_void_p]
isl.isl_basic_set_apply.restype = c_void_p
isl.isl_basic_set_apply.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_detect_equalities.restype = c_void_p
isl.isl_basic_set_detect_equalities.argtypes = [c_void_p]
isl.isl_basic_set_dim_max_val.restype = c_void_p
isl.isl_basic_set_dim_max_val.argtypes = [c_void_p, c_int]
isl.isl_basic_set_flatten.restype = c_void_p
isl.isl_basic_set_flatten.argtypes = [c_void_p]
isl.isl_basic_set_gist.restype = c_void_p
isl.isl_basic_set_gist.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_intersect.restype = c_void_p
isl.isl_basic_set_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_intersect_params.restype = c_void_p
isl.isl_basic_set_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_is_empty.argtypes = [c_void_p]
isl.isl_basic_set_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_is_wrapping.argtypes = [c_void_p]
isl.isl_basic_set_lexmax.restype = c_void_p
isl.isl_basic_set_lexmax.argtypes = [c_void_p]
isl.isl_basic_set_lexmin.restype = c_void_p
isl.isl_basic_set_lexmin.argtypes = [c_void_p]
isl.isl_basic_set_params.restype = c_void_p
isl.isl_basic_set_params.argtypes = [c_void_p]
isl.isl_basic_set_sample.restype = c_void_p
isl.isl_basic_set_sample.argtypes = [c_void_p]
isl.isl_basic_set_sample_point.restype = c_void_p
isl.isl_basic_set_sample_point.argtypes = [c_void_p]
isl.isl_basic_set_union.restype = c_void_p
isl.isl_basic_set_union.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_copy.restype = c_void_p
isl.isl_basic_set_copy.argtypes = [c_void_p]
isl.isl_basic_set_free.restype = c_void_p
isl.isl_basic_set_free.argtypes = [c_void_p]
isl.isl_basic_set_to_str.restype = POINTER(c_char)
isl.isl_basic_set_to_str.argtypes = [c_void_p]

class fixed_box(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_fixed_box_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is fixed_box:
                arg0 = fixed_box(arg0)
        except:
            raise
        ptr = isl.isl_fixed_box_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.fixed_box("""%s""")' % s
        else:
            return 'isl.fixed_box("%s")' % s
    def offset(arg0):
        try:
            if not arg0.__class__ is fixed_box:
                arg0 = fixed_box(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_fixed_box_get_offset(arg0.ptr)
        obj = multi_aff(ctx=ctx, ptr=res)
        return obj
    def get_offset(arg0):
        return arg0.offset()
    def size(arg0):
        try:
            if not arg0.__class__ is fixed_box:
                arg0 = fixed_box(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_fixed_box_get_size(arg0.ptr)
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def get_size(arg0):
        return arg0.size()
    def space(arg0):
        try:
            if not arg0.__class__ is fixed_box:
                arg0 = fixed_box(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_fixed_box_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def is_valid(arg0):
        try:
            if not arg0.__class__ is fixed_box:
                arg0 = fixed_box(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_fixed_box_is_valid(arg0.ptr)
        if res < 0:
            raise
        return bool(res)

isl.isl_fixed_box_get_offset.restype = c_void_p
isl.isl_fixed_box_get_offset.argtypes = [c_void_p]
isl.isl_fixed_box_get_size.restype = c_void_p
isl.isl_fixed_box_get_size.argtypes = [c_void_p]
isl.isl_fixed_box_get_space.restype = c_void_p
isl.isl_fixed_box_get_space.argtypes = [c_void_p]
isl.isl_fixed_box_is_valid.argtypes = [c_void_p]
isl.isl_fixed_box_copy.restype = c_void_p
isl.isl_fixed_box_copy.argtypes = [c_void_p]
isl.isl_fixed_box_free.restype = c_void_p
isl.isl_fixed_box_free.argtypes = [c_void_p]
isl.isl_fixed_box_to_str.restype = POINTER(c_char)
isl.isl_fixed_box_to_str.argtypes = [c_void_p]

class id(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_id_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_id_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is id:
                arg0 = id(arg0)
        except:
            raise
        ptr = isl.isl_id_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.id("""%s""")' % s
        else:
            return 'isl.id("%s")' % s
    def name(arg0):
        try:
            if not arg0.__class__ is id:
                arg0 = id(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_get_name(arg0.ptr)
        if res == 0:
            raise
        string = cast(res, c_char_p).value.decode('ascii')
        return string
    def get_name(arg0):
        return arg0.name()

isl.isl_id_read_from_str.restype = c_void_p
isl.isl_id_read_from_str.argtypes = [Context, c_char_p]
isl.isl_id_get_name.restype = POINTER(c_char)
isl.isl_id_get_name.argtypes = [c_void_p]
isl.isl_id_copy.restype = c_void_p
isl.isl_id_copy.argtypes = [c_void_p]
isl.isl_id_free.restype = c_void_p
isl.isl_id_free.argtypes = [c_void_p]
isl.isl_id_to_str.restype = POINTER(c_char)
isl.isl_id_to_str.argtypes = [c_void_p]

class id_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_id_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and (args[0].__class__ is id or type(args[0]) == str):
            args = list(args)
            try:
                if not args[0].__class__ is id:
                    args[0] = id(args[0])
            except:
                raise
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_id_list_from_id(isl.isl_id_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_id_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        ptr = isl.isl_id_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.id_list("""%s""")' % s
        else:
            return 'isl.id_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is id:
                arg1 = id(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_list_add(isl.isl_id_list_copy(arg0.ptr), isl.isl_id_copy(arg1.ptr))
        obj = id_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_list_clear(isl.isl_id_list_copy(arg0.ptr))
        obj = id_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is id_list:
                arg1 = id_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_list_concat(isl.isl_id_list_copy(arg0.ptr), isl.isl_id_list_copy(arg1.ptr))
        obj = id_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_list_drop(isl.isl_id_list_copy(arg0.ptr), arg1, arg2)
        obj = id_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = id(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_id_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_list_get_at(arg0.ptr, arg1)
        obj = id(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is id:
                arg2 = id(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_list_insert(isl.isl_id_list_copy(arg0.ptr), arg1, isl.isl_id_copy(arg2.ptr))
        obj = id_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is id_list:
                arg0 = id_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_id_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_id_list_alloc.restype = c_void_p
isl.isl_id_list_alloc.argtypes = [Context, c_int]
isl.isl_id_list_from_id.restype = c_void_p
isl.isl_id_list_from_id.argtypes = [c_void_p]
isl.isl_id_list_add.restype = c_void_p
isl.isl_id_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_id_list_clear.restype = c_void_p
isl.isl_id_list_clear.argtypes = [c_void_p]
isl.isl_id_list_concat.restype = c_void_p
isl.isl_id_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_id_list_drop.restype = c_void_p
isl.isl_id_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_id_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_id_list_get_at.restype = c_void_p
isl.isl_id_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_id_list_insert.restype = c_void_p
isl.isl_id_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_id_list_size.argtypes = [c_void_p]
isl.isl_id_list_copy.restype = c_void_p
isl.isl_id_list_copy.argtypes = [c_void_p]
isl.isl_id_list_free.restype = c_void_p
isl.isl_id_list_free.argtypes = [c_void_p]
isl.isl_id_list_to_str.restype = POINTER(c_char)
isl.isl_id_list_to_str.argtypes = [c_void_p]

class multi_id(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 2 and args[0].__class__ is space and args[1].__class__ is id_list:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_id_from_id_list(isl.isl_space_copy(args[0].ptr), isl.isl_id_list_copy(args[1].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_id_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_id_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        ptr = isl.isl_multi_id_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_id("""%s""")' % s
        else:
            return 'isl.multi_id("%s")' % s
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_flat_range_product(isl.isl_multi_id_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_id(ctx=ctx, ptr=res)
        return obj
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_get_at(arg0.ptr, arg1)
        obj = id(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def list(arg0):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_get_list(arg0.ptr)
        obj = id_list(ctx=ctx, ptr=res)
        return obj
    def get_list(arg0):
        return arg0.list()
    def space(arg0):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def plain_is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_plain_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_id:
                arg1 = multi_id(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_range_product(isl.isl_multi_id_copy(arg0.ptr), isl.isl_multi_id_copy(arg1.ptr))
        obj = multi_id(ctx=ctx, ptr=res)
        return obj
    def set_at(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is id:
                arg2 = id(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_set_at(isl.isl_multi_id_copy(arg0.ptr), arg1, isl.isl_id_copy(arg2.ptr))
        obj = multi_id(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is multi_id:
                arg0 = multi_id(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_id_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_multi_id_from_id_list.restype = c_void_p
isl.isl_multi_id_from_id_list.argtypes = [c_void_p, c_void_p]
isl.isl_multi_id_read_from_str.restype = c_void_p
isl.isl_multi_id_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_id_flat_range_product.restype = c_void_p
isl.isl_multi_id_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_id_get_at.restype = c_void_p
isl.isl_multi_id_get_at.argtypes = [c_void_p, c_int]
isl.isl_multi_id_get_list.restype = c_void_p
isl.isl_multi_id_get_list.argtypes = [c_void_p]
isl.isl_multi_id_get_space.restype = c_void_p
isl.isl_multi_id_get_space.argtypes = [c_void_p]
isl.isl_multi_id_plain_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_multi_id_range_product.restype = c_void_p
isl.isl_multi_id_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_id_set_at.restype = c_void_p
isl.isl_multi_id_set_at.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_multi_id_size.argtypes = [c_void_p]
isl.isl_multi_id_copy.restype = c_void_p
isl.isl_multi_id_copy.argtypes = [c_void_p]
isl.isl_multi_id_free.restype = c_void_p
isl.isl_multi_id_free.argtypes = [c_void_p]
isl.isl_multi_id_to_str.restype = POINTER(c_char)
isl.isl_multi_id_to_str.argtypes = [c_void_p]

class multi_val(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 2 and args[0].__class__ is space and args[1].__class__ is val_list:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_val_from_val_list(isl.isl_space_copy(args[0].ptr), isl.isl_val_list_copy(args[1].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_val_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_val_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ptr = isl.isl_multi_val_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_val("""%s""")' % s
        else:
            return 'isl.multi_val("%s")' % s
    def add(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_val_add(isl.isl_multi_val_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_val(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_val_add_val(isl.isl_multi_val_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_val(ctx=ctx, ptr=res)
            return obj
        raise Error
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_flat_range_product(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_get_at(arg0.ptr, arg1)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def list(arg0):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_get_list(arg0.ptr)
        obj = val_list(ctx=ctx, ptr=res)
        return obj
    def get_list(arg0):
        return arg0.list()
    def space(arg0):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_get_space(arg0.ptr)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def get_space(arg0):
        return arg0.space()
    def involves_nan(arg0):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_involves_nan(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def max(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_max(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def min(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_min(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def neg(arg0):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_neg(isl.isl_multi_val_copy(arg0.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def plain_is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_plain_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_product(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_range_product(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def scale(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_val_scale_multi_val(isl.isl_multi_val_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_val(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_val_scale_val(isl.isl_multi_val_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_val(ctx=ctx, ptr=res)
            return obj
        raise Error
    def scale_down(*args):
        if len(args) == 2 and args[1].__class__ is multi_val:
            ctx = args[0].ctx
            res = isl.isl_multi_val_scale_down_multi_val(isl.isl_multi_val_copy(args[0].ptr), isl.isl_multi_val_copy(args[1].ptr))
            obj = multi_val(ctx=ctx, ptr=res)
            return obj
        if len(args) == 2 and (args[1].__class__ is val or type(args[1]) == int):
            args = list(args)
            try:
                if not args[1].__class__ is val:
                    args[1] = val(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_multi_val_scale_down_val(isl.isl_multi_val_copy(args[0].ptr), isl.isl_val_copy(args[1].ptr))
            obj = multi_val(ctx=ctx, ptr=res)
            return obj
        raise Error
    def set_at(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is val:
                arg2 = val(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_set_at(isl.isl_multi_val_copy(arg0.ptr), arg1, isl.isl_val_copy(arg2.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_sub(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def zero(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_zero(isl.isl_space_copy(arg0.ptr))
        obj = multi_val(ctx=ctx, ptr=res)
        return obj

isl.isl_multi_val_from_val_list.restype = c_void_p
isl.isl_multi_val_from_val_list.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_read_from_str.restype = c_void_p
isl.isl_multi_val_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_val_add.restype = c_void_p
isl.isl_multi_val_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_add_val.restype = c_void_p
isl.isl_multi_val_add_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_flat_range_product.restype = c_void_p
isl.isl_multi_val_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_get_at.restype = c_void_p
isl.isl_multi_val_get_at.argtypes = [c_void_p, c_int]
isl.isl_multi_val_get_list.restype = c_void_p
isl.isl_multi_val_get_list.argtypes = [c_void_p]
isl.isl_multi_val_get_space.restype = c_void_p
isl.isl_multi_val_get_space.argtypes = [c_void_p]
isl.isl_multi_val_involves_nan.argtypes = [c_void_p]
isl.isl_multi_val_max.restype = c_void_p
isl.isl_multi_val_max.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_min.restype = c_void_p
isl.isl_multi_val_min.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_neg.restype = c_void_p
isl.isl_multi_val_neg.argtypes = [c_void_p]
isl.isl_multi_val_plain_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_product.restype = c_void_p
isl.isl_multi_val_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_range_product.restype = c_void_p
isl.isl_multi_val_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_scale_multi_val.restype = c_void_p
isl.isl_multi_val_scale_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_scale_val.restype = c_void_p
isl.isl_multi_val_scale_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_scale_down_multi_val.restype = c_void_p
isl.isl_multi_val_scale_down_multi_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_scale_down_val.restype = c_void_p
isl.isl_multi_val_scale_down_val.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_set_at.restype = c_void_p
isl.isl_multi_val_set_at.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_multi_val_size.argtypes = [c_void_p]
isl.isl_multi_val_sub.restype = c_void_p
isl.isl_multi_val_sub.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_zero.restype = c_void_p
isl.isl_multi_val_zero.argtypes = [c_void_p]
isl.isl_multi_val_copy.restype = c_void_p
isl.isl_multi_val_copy.argtypes = [c_void_p]
isl.isl_multi_val_free.restype = c_void_p
isl.isl_multi_val_free.argtypes = [c_void_p]
isl.isl_multi_val_to_str.restype = POINTER(c_char)
isl.isl_multi_val_to_str.argtypes = [c_void_p]

class point(basic_set):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_point_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is point:
                arg0 = point(arg0)
        except:
            raise
        ptr = isl.isl_point_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.point("""%s""")' % s
        else:
            return 'isl.point("%s")' % s
    def multi_val(arg0):
        try:
            if not arg0.__class__ is point:
                arg0 = point(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_point_get_multi_val(arg0.ptr)
        obj = multi_val(ctx=ctx, ptr=res)
        return obj
    def get_multi_val(arg0):
        return arg0.multi_val()

isl.isl_point_get_multi_val.restype = c_void_p
isl.isl_point_get_multi_val.argtypes = [c_void_p]
isl.isl_point_copy.restype = c_void_p
isl.isl_point_copy.argtypes = [c_void_p]
isl.isl_point_free.restype = c_void_p
isl.isl_point_free.argtypes = [c_void_p]
isl.isl_point_to_str.restype = POINTER(c_char)
isl.isl_point_to_str.argtypes = [c_void_p]

class pw_aff_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_aff_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_aff_list_from_pw_aff(isl.isl_pw_aff_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_pw_aff_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        ptr = isl.isl_pw_aff_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.pw_aff_list("""%s""")' % s
        else:
            return 'isl.pw_aff_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_add(isl.isl_pw_aff_list_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        obj = pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_clear(isl.isl_pw_aff_list_copy(arg0.ptr))
        obj = pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff_list:
                arg1 = pw_aff_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_concat(isl.isl_pw_aff_list_copy(arg0.ptr), isl.isl_pw_aff_list_copy(arg1.ptr))
        obj = pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_drop(isl.isl_pw_aff_list_copy(arg0.ptr), arg1, arg2)
        obj = pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = pw_aff(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_get_at(arg0.ptr, arg1)
        obj = pw_aff(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is pw_aff:
                arg2 = pw_aff(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_insert(isl.isl_pw_aff_list_copy(arg0.ptr), arg1, isl.isl_pw_aff_copy(arg2.ptr))
        obj = pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is pw_aff_list:
                arg0 = pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_aff_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_pw_aff_list_alloc.restype = c_void_p
isl.isl_pw_aff_list_alloc.argtypes = [Context, c_int]
isl.isl_pw_aff_list_from_pw_aff.restype = c_void_p
isl.isl_pw_aff_list_from_pw_aff.argtypes = [c_void_p]
isl.isl_pw_aff_list_add.restype = c_void_p
isl.isl_pw_aff_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_list_clear.restype = c_void_p
isl.isl_pw_aff_list_clear.argtypes = [c_void_p]
isl.isl_pw_aff_list_concat.restype = c_void_p
isl.isl_pw_aff_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_list_drop.restype = c_void_p
isl.isl_pw_aff_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_pw_aff_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_pw_aff_list_get_at.restype = c_void_p
isl.isl_pw_aff_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_pw_aff_list_insert.restype = c_void_p
isl.isl_pw_aff_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_pw_aff_list_size.argtypes = [c_void_p]
isl.isl_pw_aff_list_copy.restype = c_void_p
isl.isl_pw_aff_list_copy.argtypes = [c_void_p]
isl.isl_pw_aff_list_free.restype = c_void_p
isl.isl_pw_aff_list_free.argtypes = [c_void_p]
isl.isl_pw_aff_list_to_str.restype = POINTER(c_char)
isl.isl_pw_aff_list_to_str.argtypes = [c_void_p]

class pw_multi_aff_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is pw_multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_list_from_pw_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_pw_multi_aff_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        ptr = isl.isl_pw_multi_aff_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.pw_multi_aff_list("""%s""")' % s
        else:
            return 'isl.pw_multi_aff_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_add(isl.isl_pw_multi_aff_list_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        obj = pw_multi_aff_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_clear(isl.isl_pw_multi_aff_list_copy(arg0.ptr))
        obj = pw_multi_aff_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff_list:
                arg1 = pw_multi_aff_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_concat(isl.isl_pw_multi_aff_list_copy(arg0.ptr), isl.isl_pw_multi_aff_list_copy(arg1.ptr))
        obj = pw_multi_aff_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_drop(isl.isl_pw_multi_aff_list_copy(arg0.ptr), arg1, arg2)
        obj = pw_multi_aff_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = pw_multi_aff(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_get_at(arg0.ptr, arg1)
        obj = pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is pw_multi_aff:
                arg2 = pw_multi_aff(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_insert(isl.isl_pw_multi_aff_list_copy(arg0.ptr), arg1, isl.isl_pw_multi_aff_copy(arg2.ptr))
        obj = pw_multi_aff_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff_list:
                arg0 = pw_multi_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_pw_multi_aff_list_alloc.restype = c_void_p
isl.isl_pw_multi_aff_list_alloc.argtypes = [Context, c_int]
isl.isl_pw_multi_aff_list_from_pw_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_list_from_pw_multi_aff.argtypes = [c_void_p]
isl.isl_pw_multi_aff_list_add.restype = c_void_p
isl.isl_pw_multi_aff_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_list_clear.restype = c_void_p
isl.isl_pw_multi_aff_list_clear.argtypes = [c_void_p]
isl.isl_pw_multi_aff_list_concat.restype = c_void_p
isl.isl_pw_multi_aff_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_list_drop.restype = c_void_p
isl.isl_pw_multi_aff_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_pw_multi_aff_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_pw_multi_aff_list_get_at.restype = c_void_p
isl.isl_pw_multi_aff_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_pw_multi_aff_list_insert.restype = c_void_p
isl.isl_pw_multi_aff_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_pw_multi_aff_list_size.argtypes = [c_void_p]
isl.isl_pw_multi_aff_list_copy.restype = c_void_p
isl.isl_pw_multi_aff_list_copy.argtypes = [c_void_p]
isl.isl_pw_multi_aff_list_free.restype = c_void_p
isl.isl_pw_multi_aff_list_free.argtypes = [c_void_p]
isl.isl_pw_multi_aff_list_to_str.restype = POINTER(c_char)
isl.isl_pw_multi_aff_list_to_str.argtypes = [c_void_p]

class schedule(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_schedule_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule:
                arg0 = schedule(arg0)
        except:
            raise
        ptr = isl.isl_schedule_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule("""%s""")' % s
        else:
            return 'isl.schedule("%s")' % s
    @staticmethod
    def from_domain(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_from_domain(isl.isl_union_set_copy(arg0.ptr))
        obj = schedule(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is schedule:
                arg0 = schedule(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_get_domain(arg0.ptr)
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def get_domain(arg0):
        return arg0.domain()
    def map(arg0):
        try:
            if not arg0.__class__ is schedule:
                arg0 = schedule(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_get_map(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_map(arg0):
        return arg0.map()
    def root(arg0):
        try:
            if not arg0.__class__ is schedule:
                arg0 = schedule(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_get_root(arg0.ptr)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def get_root(arg0):
        return arg0.root()
    def pullback(*args):
        if len(args) == 2 and args[1].__class__ is union_pw_multi_aff:
            ctx = args[0].ctx
            res = isl.isl_schedule_pullback_union_pw_multi_aff(isl.isl_schedule_copy(args[0].ptr), isl.isl_union_pw_multi_aff_copy(args[1].ptr))
            obj = schedule(ctx=ctx, ptr=res)
            return obj
        raise Error

isl.isl_schedule_read_from_str.restype = c_void_p
isl.isl_schedule_read_from_str.argtypes = [Context, c_char_p]
isl.isl_schedule_from_domain.restype = c_void_p
isl.isl_schedule_from_domain.argtypes = [c_void_p]
isl.isl_schedule_get_domain.restype = c_void_p
isl.isl_schedule_get_domain.argtypes = [c_void_p]
isl.isl_schedule_get_map.restype = c_void_p
isl.isl_schedule_get_map.argtypes = [c_void_p]
isl.isl_schedule_get_root.restype = c_void_p
isl.isl_schedule_get_root.argtypes = [c_void_p]
isl.isl_schedule_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_schedule_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_copy.restype = c_void_p
isl.isl_schedule_copy.argtypes = [c_void_p]
isl.isl_schedule_free.restype = c_void_p
isl.isl_schedule_free.argtypes = [c_void_p]
isl.isl_schedule_to_str.restype = POINTER(c_char)
isl.isl_schedule_to_str.argtypes = [c_void_p]

class schedule_constraints(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_schedule_constraints_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_constraints_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ptr = isl.isl_schedule_constraints_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_constraints("""%s""")' % s
        else:
            return 'isl.schedule_constraints("%s")' % s
    def compute_schedule(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_compute_schedule(isl.isl_schedule_constraints_copy(arg0.ptr))
        obj = schedule(ctx=ctx, ptr=res)
        return obj
    def coincidence(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_coincidence(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_coincidence(arg0):
        return arg0.coincidence()
    def conditional_validity(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_conditional_validity(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_conditional_validity(arg0):
        return arg0.conditional_validity()
    def conditional_validity_condition(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_conditional_validity_condition(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_conditional_validity_condition(arg0):
        return arg0.conditional_validity_condition()
    def context(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_context(arg0.ptr)
        obj = set(ctx=ctx, ptr=res)
        return obj
    def get_context(arg0):
        return arg0.context()
    def domain(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_domain(arg0.ptr)
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def get_domain(arg0):
        return arg0.domain()
    def proximity(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_proximity(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_proximity(arg0):
        return arg0.proximity()
    def validity(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_validity(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_validity(arg0):
        return arg0.validity()
    @staticmethod
    def on_domain(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_on_domain(isl.isl_union_set_copy(arg0.ptr))
        obj = schedule_constraints(ctx=ctx, ptr=res)
        return obj
    def set_coincidence(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_set_coincidence(isl.isl_schedule_constraints_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = schedule_constraints(ctx=ctx, ptr=res)
        return obj
    def set_conditional_validity(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        try:
            if not arg2.__class__ is union_map:
                arg2 = union_map(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_set_conditional_validity(isl.isl_schedule_constraints_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr), isl.isl_union_map_copy(arg2.ptr))
        obj = schedule_constraints(ctx=ctx, ptr=res)
        return obj
    def set_context(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_set_context(isl.isl_schedule_constraints_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = schedule_constraints(ctx=ctx, ptr=res)
        return obj
    def set_proximity(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_set_proximity(isl.isl_schedule_constraints_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = schedule_constraints(ctx=ctx, ptr=res)
        return obj
    def set_validity(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_set_validity(isl.isl_schedule_constraints_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = schedule_constraints(ctx=ctx, ptr=res)
        return obj

isl.isl_schedule_constraints_read_from_str.restype = c_void_p
isl.isl_schedule_constraints_read_from_str.argtypes = [Context, c_char_p]
isl.isl_schedule_constraints_compute_schedule.restype = c_void_p
isl.isl_schedule_constraints_compute_schedule.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_coincidence.restype = c_void_p
isl.isl_schedule_constraints_get_coincidence.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_conditional_validity.restype = c_void_p
isl.isl_schedule_constraints_get_conditional_validity.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_conditional_validity_condition.restype = c_void_p
isl.isl_schedule_constraints_get_conditional_validity_condition.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_context.restype = c_void_p
isl.isl_schedule_constraints_get_context.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_domain.restype = c_void_p
isl.isl_schedule_constraints_get_domain.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_proximity.restype = c_void_p
isl.isl_schedule_constraints_get_proximity.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_validity.restype = c_void_p
isl.isl_schedule_constraints_get_validity.argtypes = [c_void_p]
isl.isl_schedule_constraints_on_domain.restype = c_void_p
isl.isl_schedule_constraints_on_domain.argtypes = [c_void_p]
isl.isl_schedule_constraints_set_coincidence.restype = c_void_p
isl.isl_schedule_constraints_set_coincidence.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_constraints_set_conditional_validity.restype = c_void_p
isl.isl_schedule_constraints_set_conditional_validity.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_schedule_constraints_set_context.restype = c_void_p
isl.isl_schedule_constraints_set_context.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_constraints_set_proximity.restype = c_void_p
isl.isl_schedule_constraints_set_proximity.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_constraints_set_validity.restype = c_void_p
isl.isl_schedule_constraints_set_validity.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_constraints_copy.restype = c_void_p
isl.isl_schedule_constraints_copy.argtypes = [c_void_p]
isl.isl_schedule_constraints_free.restype = c_void_p
isl.isl_schedule_constraints_free.argtypes = [c_void_p]
isl.isl_schedule_constraints_to_str.restype = POINTER(c_char)
isl.isl_schedule_constraints_to_str.argtypes = [c_void_p]

class schedule_node(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_band):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_context):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_domain):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_expansion):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_extension):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_filter):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_leaf):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_guard):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_mark):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_sequence):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        if len(args) == 1 and isinstance(args[0], schedule_node_set):
            self.ctx = args[0].ctx
            self.ptr = isl.isl_schedule_node_copy(args[0].ptr)
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        if "ptr" in keywords:
            type = isl.isl_schedule_node_get_type(keywords["ptr"])
            if type == 0:
                return schedule_node_band(**keywords)
            if type == 1:
                return schedule_node_context(**keywords)
            if type == 2:
                return schedule_node_domain(**keywords)
            if type == 3:
                return schedule_node_expansion(**keywords)
            if type == 4:
                return schedule_node_extension(**keywords)
            if type == 5:
                return schedule_node_filter(**keywords)
            if type == 6:
                return schedule_node_leaf(**keywords)
            if type == 7:
                return schedule_node_guard(**keywords)
            if type == 8:
                return schedule_node_mark(**keywords)
            if type == 9:
                return schedule_node_sequence(**keywords)
            if type == 10:
                return schedule_node_set(**keywords)
            raise
        return super(schedule_node, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node("""%s""")' % s
        else:
            return 'isl.schedule_node("%s")' % s
    def ancestor(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_ancestor(isl.isl_schedule_node_copy(arg0.ptr), arg1)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def child(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_child(isl.isl_schedule_node_copy(arg0.ptr), arg1)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def every_descendant(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = schedule_node(ctx=arg0.ctx, ptr=isl.isl_schedule_node_copy(cb_arg0))
            try:
                res = arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 1 if res else 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_schedule_node_every_descendant(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
        return bool(res)
    def first_child(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_first_child(isl.isl_schedule_node_copy(arg0.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def foreach_ancestor_top_down(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = schedule_node(ctx=arg0.ctx, ptr=isl.isl_schedule_node_copy(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_schedule_node_foreach_ancestor_top_down(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def foreach_descendant_top_down(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = schedule_node(ctx=arg0.ctx, ptr=isl.isl_schedule_node_copy(cb_arg0))
            try:
                res = arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 1 if res else 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_schedule_node_foreach_descendant_top_down(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    @staticmethod
    def from_domain(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_from_domain(isl.isl_union_set_copy(arg0.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def from_extension(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_from_extension(isl.isl_union_map_copy(arg0.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def ancestor_child_position(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is schedule_node:
                arg1 = schedule_node(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_ancestor_child_position(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return int(res)
    def get_ancestor_child_position(arg0, arg1):
        return arg0.ancestor_child_position(arg1)
    def child_position(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_child_position(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def get_child_position(arg0):
        return arg0.child_position()
    def prefix_schedule_multi_union_pw_aff(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(arg0.ptr)
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def get_prefix_schedule_multi_union_pw_aff(arg0):
        return arg0.prefix_schedule_multi_union_pw_aff()
    def prefix_schedule_union_map(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_prefix_schedule_union_map(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_prefix_schedule_union_map(arg0):
        return arg0.prefix_schedule_union_map()
    def prefix_schedule_union_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(arg0.ptr)
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def get_prefix_schedule_union_pw_multi_aff(arg0):
        return arg0.prefix_schedule_union_pw_multi_aff()
    def schedule(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_schedule(arg0.ptr)
        obj = schedule(ctx=ctx, ptr=res)
        return obj
    def get_schedule(arg0):
        return arg0.schedule()
    def shared_ancestor(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is schedule_node:
                arg1 = schedule_node(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_shared_ancestor(arg0.ptr, arg1.ptr)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def get_shared_ancestor(arg0, arg1):
        return arg0.shared_ancestor(arg1)
    def tree_depth(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_tree_depth(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def get_tree_depth(arg0):
        return arg0.tree_depth()
    def graft_after(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is schedule_node:
                arg1 = schedule_node(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_graft_after(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_schedule_node_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def graft_before(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is schedule_node:
                arg1 = schedule_node(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_graft_before(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_schedule_node_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def has_children(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_has_children(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def has_next_sibling(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_has_next_sibling(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def has_parent(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_has_parent(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def has_previous_sibling(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_has_previous_sibling(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def insert_context(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_insert_context(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def insert_filter(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_insert_filter(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def insert_guard(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_insert_guard(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def insert_mark(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is id:
                arg1 = id(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_insert_mark(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_id_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def insert_partial_schedule(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_insert_partial_schedule(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def insert_sequence(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set_list:
                arg1 = union_set_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_insert_sequence(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_union_set_list_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def insert_set(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set_list:
                arg1 = union_set_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_insert_set(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_union_set_list_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is schedule_node:
                arg1 = schedule_node(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subtree_anchored(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_is_subtree_anchored(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def map_descendant_bottom_up(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_void_p, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = schedule_node(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                res = arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return None
            return isl.isl_schedule_node_copy(res.ptr)
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_schedule_node_map_descendant_bottom_up(isl.isl_schedule_node_copy(arg0.ptr), cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def n_children(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_n_children(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def next_sibling(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_next_sibling(isl.isl_schedule_node_copy(arg0.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def order_after(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_order_after(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def order_before(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_order_before(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def parent(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_parent(isl.isl_schedule_node_copy(arg0.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def previous_sibling(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_previous_sibling(isl.isl_schedule_node_copy(arg0.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def root(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_root(isl.isl_schedule_node_copy(arg0.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj

isl.isl_schedule_node_ancestor.restype = c_void_p
isl.isl_schedule_node_ancestor.argtypes = [c_void_p, c_int]
isl.isl_schedule_node_child.restype = c_void_p
isl.isl_schedule_node_child.argtypes = [c_void_p, c_int]
isl.isl_schedule_node_every_descendant.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_schedule_node_first_child.restype = c_void_p
isl.isl_schedule_node_first_child.argtypes = [c_void_p]
isl.isl_schedule_node_foreach_ancestor_top_down.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_schedule_node_foreach_descendant_top_down.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_schedule_node_from_domain.restype = c_void_p
isl.isl_schedule_node_from_domain.argtypes = [c_void_p]
isl.isl_schedule_node_from_extension.restype = c_void_p
isl.isl_schedule_node_from_extension.argtypes = [c_void_p]
isl.isl_schedule_node_get_ancestor_child_position.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_get_child_position.argtypes = [c_void_p]
isl.isl_schedule_node_get_prefix_schedule_multi_union_pw_aff.restype = c_void_p
isl.isl_schedule_node_get_prefix_schedule_multi_union_pw_aff.argtypes = [c_void_p]
isl.isl_schedule_node_get_prefix_schedule_union_map.restype = c_void_p
isl.isl_schedule_node_get_prefix_schedule_union_map.argtypes = [c_void_p]
isl.isl_schedule_node_get_prefix_schedule_union_pw_multi_aff.restype = c_void_p
isl.isl_schedule_node_get_prefix_schedule_union_pw_multi_aff.argtypes = [c_void_p]
isl.isl_schedule_node_get_schedule.restype = c_void_p
isl.isl_schedule_node_get_schedule.argtypes = [c_void_p]
isl.isl_schedule_node_get_shared_ancestor.restype = c_void_p
isl.isl_schedule_node_get_shared_ancestor.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_get_tree_depth.argtypes = [c_void_p]
isl.isl_schedule_node_graft_after.restype = c_void_p
isl.isl_schedule_node_graft_after.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_graft_before.restype = c_void_p
isl.isl_schedule_node_graft_before.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_has_children.argtypes = [c_void_p]
isl.isl_schedule_node_has_next_sibling.argtypes = [c_void_p]
isl.isl_schedule_node_has_parent.argtypes = [c_void_p]
isl.isl_schedule_node_has_previous_sibling.argtypes = [c_void_p]
isl.isl_schedule_node_insert_context.restype = c_void_p
isl.isl_schedule_node_insert_context.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_insert_filter.restype = c_void_p
isl.isl_schedule_node_insert_filter.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_insert_guard.restype = c_void_p
isl.isl_schedule_node_insert_guard.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_insert_mark.restype = c_void_p
isl.isl_schedule_node_insert_mark.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_insert_partial_schedule.restype = c_void_p
isl.isl_schedule_node_insert_partial_schedule.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_insert_sequence.restype = c_void_p
isl.isl_schedule_node_insert_sequence.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_insert_set.restype = c_void_p
isl.isl_schedule_node_insert_set.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_is_subtree_anchored.argtypes = [c_void_p]
isl.isl_schedule_node_map_descendant_bottom_up.restype = c_void_p
isl.isl_schedule_node_map_descendant_bottom_up.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_schedule_node_n_children.argtypes = [c_void_p]
isl.isl_schedule_node_next_sibling.restype = c_void_p
isl.isl_schedule_node_next_sibling.argtypes = [c_void_p]
isl.isl_schedule_node_order_after.restype = c_void_p
isl.isl_schedule_node_order_after.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_order_before.restype = c_void_p
isl.isl_schedule_node_order_before.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_parent.restype = c_void_p
isl.isl_schedule_node_parent.argtypes = [c_void_p]
isl.isl_schedule_node_previous_sibling.restype = c_void_p
isl.isl_schedule_node_previous_sibling.argtypes = [c_void_p]
isl.isl_schedule_node_root.restype = c_void_p
isl.isl_schedule_node_root.argtypes = [c_void_p]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]
isl.isl_schedule_node_get_type.argtypes = [c_void_p]

class schedule_node_band(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_band, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_band:
                arg0 = schedule_node_band(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_band("""%s""")' % s
        else:
            return 'isl.schedule_node_band("%s")' % s
    def ast_build_options(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_get_ast_build_options(arg0.ptr)
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def get_ast_build_options(arg0):
        return arg0.ast_build_options()
    def ast_isolate_option(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_get_ast_isolate_option(arg0.ptr)
        obj = set(ctx=ctx, ptr=res)
        return obj
    def get_ast_isolate_option(arg0):
        return arg0.ast_isolate_option()
    def partial_schedule(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_get_partial_schedule(arg0.ptr)
        obj = multi_union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def get_partial_schedule(arg0):
        return arg0.partial_schedule()
    def permutable(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_get_permutable(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def get_permutable(arg0):
        return arg0.permutable()
    def member_get_coincident(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_get_coincident(arg0.ptr, arg1)
        if res < 0:
            raise
        return bool(res)
    def member_set_coincident(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_set_coincident(isl.isl_schedule_node_copy(arg0.ptr), arg1, arg2)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def mod(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_mod(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def n_member(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_n_member(arg0.ptr)
        if res < 0:
            raise
        return int(res)
    def scale(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_scale(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def scale_down(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_scale_down(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def set_ast_build_options(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_set_ast_build_options(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def set_permutable(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_set_permutable(isl.isl_schedule_node_copy(arg0.ptr), arg1)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def shift(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_shift(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def split(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_split(isl.isl_schedule_node_copy(arg0.ptr), arg1)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def tile(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_tile(isl.isl_schedule_node_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def member_set_ast_loop_default(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_set_ast_loop_type(isl.isl_schedule_node_copy(arg0.ptr), arg1, 0)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def member_set_ast_loop_atomic(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_set_ast_loop_type(isl.isl_schedule_node_copy(arg0.ptr), arg1, 1)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def member_set_ast_loop_unroll(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_set_ast_loop_type(isl.isl_schedule_node_copy(arg0.ptr), arg1, 2)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj
    def member_set_ast_loop_separate(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_set_ast_loop_type(isl.isl_schedule_node_copy(arg0.ptr), arg1, 3)
        obj = schedule_node(ctx=ctx, ptr=res)
        return obj

isl.isl_schedule_node_band_get_ast_build_options.restype = c_void_p
isl.isl_schedule_node_band_get_ast_build_options.argtypes = [c_void_p]
isl.isl_schedule_node_band_get_ast_isolate_option.restype = c_void_p
isl.isl_schedule_node_band_get_ast_isolate_option.argtypes = [c_void_p]
isl.isl_schedule_node_band_get_partial_schedule.restype = c_void_p
isl.isl_schedule_node_band_get_partial_schedule.argtypes = [c_void_p]
isl.isl_schedule_node_band_get_permutable.argtypes = [c_void_p]
isl.isl_schedule_node_band_member_get_coincident.argtypes = [c_void_p, c_int]
isl.isl_schedule_node_band_member_set_coincident.restype = c_void_p
isl.isl_schedule_node_band_member_set_coincident.argtypes = [c_void_p, c_int, c_int]
isl.isl_schedule_node_band_mod.restype = c_void_p
isl.isl_schedule_node_band_mod.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_band_n_member.argtypes = [c_void_p]
isl.isl_schedule_node_band_scale.restype = c_void_p
isl.isl_schedule_node_band_scale.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_band_scale_down.restype = c_void_p
isl.isl_schedule_node_band_scale_down.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_band_set_ast_build_options.restype = c_void_p
isl.isl_schedule_node_band_set_ast_build_options.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_band_set_permutable.restype = c_void_p
isl.isl_schedule_node_band_set_permutable.argtypes = [c_void_p, c_int]
isl.isl_schedule_node_band_shift.restype = c_void_p
isl.isl_schedule_node_band_shift.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_band_split.restype = c_void_p
isl.isl_schedule_node_band_split.argtypes = [c_void_p, c_int]
isl.isl_schedule_node_band_tile.restype = c_void_p
isl.isl_schedule_node_band_tile.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_node_band_member_set_ast_loop_type.restype = c_void_p
isl.isl_schedule_node_band_member_set_ast_loop_type.argtypes = [c_void_p, c_int, c_int]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_context(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_context, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_context:
                arg0 = schedule_node_context(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_context("""%s""")' % s
        else:
            return 'isl.schedule_node_context("%s")' % s
    def context(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_context_get_context(arg0.ptr)
        obj = set(ctx=ctx, ptr=res)
        return obj
    def get_context(arg0):
        return arg0.context()

isl.isl_schedule_node_context_get_context.restype = c_void_p
isl.isl_schedule_node_context_get_context.argtypes = [c_void_p]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_domain(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_domain, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_domain:
                arg0 = schedule_node_domain(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_domain("""%s""")' % s
        else:
            return 'isl.schedule_node_domain("%s")' % s
    def domain(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_domain_get_domain(arg0.ptr)
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def get_domain(arg0):
        return arg0.domain()

isl.isl_schedule_node_domain_get_domain.restype = c_void_p
isl.isl_schedule_node_domain_get_domain.argtypes = [c_void_p]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_expansion(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_expansion, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_expansion:
                arg0 = schedule_node_expansion(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_expansion("""%s""")' % s
        else:
            return 'isl.schedule_node_expansion("%s")' % s
    def contraction(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_expansion_get_contraction(arg0.ptr)
        obj = union_pw_multi_aff(ctx=ctx, ptr=res)
        return obj
    def get_contraction(arg0):
        return arg0.contraction()
    def expansion(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_expansion_get_expansion(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_expansion(arg0):
        return arg0.expansion()

isl.isl_schedule_node_expansion_get_contraction.restype = c_void_p
isl.isl_schedule_node_expansion_get_contraction.argtypes = [c_void_p]
isl.isl_schedule_node_expansion_get_expansion.restype = c_void_p
isl.isl_schedule_node_expansion_get_expansion.argtypes = [c_void_p]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_extension(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_extension, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_extension:
                arg0 = schedule_node_extension(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_extension("""%s""")' % s
        else:
            return 'isl.schedule_node_extension("%s")' % s
    def extension(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_extension_get_extension(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_extension(arg0):
        return arg0.extension()

isl.isl_schedule_node_extension_get_extension.restype = c_void_p
isl.isl_schedule_node_extension_get_extension.argtypes = [c_void_p]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_filter(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_filter, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_filter:
                arg0 = schedule_node_filter(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_filter("""%s""")' % s
        else:
            return 'isl.schedule_node_filter("%s")' % s
    def filter(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_filter_get_filter(arg0.ptr)
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def get_filter(arg0):
        return arg0.filter()

isl.isl_schedule_node_filter_get_filter.restype = c_void_p
isl.isl_schedule_node_filter_get_filter.argtypes = [c_void_p]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_guard(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_guard, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_guard:
                arg0 = schedule_node_guard(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_guard("""%s""")' % s
        else:
            return 'isl.schedule_node_guard("%s")' % s
    def guard(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_guard_get_guard(arg0.ptr)
        obj = set(ctx=ctx, ptr=res)
        return obj
    def get_guard(arg0):
        return arg0.guard()

isl.isl_schedule_node_guard_get_guard.restype = c_void_p
isl.isl_schedule_node_guard_get_guard.argtypes = [c_void_p]
isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_leaf(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_leaf, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_leaf:
                arg0 = schedule_node_leaf(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_leaf("""%s""")' % s
        else:
            return 'isl.schedule_node_leaf("%s")' % s

isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_mark(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_mark, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_mark:
                arg0 = schedule_node_mark(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_mark("""%s""")' % s
        else:
            return 'isl.schedule_node_mark("%s")' % s

isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_sequence(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_sequence, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_sequence:
                arg0 = schedule_node_sequence(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_sequence("""%s""")' % s
        else:
            return 'isl.schedule_node_sequence("%s")' % s

isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class schedule_node_set(schedule_node):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __new__(cls, *args, **keywords):
        return super(schedule_node_set, cls).__new__(cls)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node_set:
                arg0 = schedule_node_set(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node_set("""%s""")' % s
        else:
            return 'isl.schedule_node_set("%s")' % s

isl.isl_schedule_node_copy.restype = c_void_p
isl.isl_schedule_node_copy.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class space(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_space_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ptr = isl.isl_space_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.space("""%s""")' % s
        else:
            return 'isl.space("%s")' % s
    def add_named_tuple(*args):
        if len(args) == 3 and (args[1].__class__ is id or type(args[1]) == str) and type(args[2]) == int:
            args = list(args)
            try:
                if not args[1].__class__ is id:
                    args[1] = id(args[1])
            except:
                raise
            ctx = args[0].ctx
            res = isl.isl_space_add_named_tuple_id_ui(isl.isl_space_copy(args[0].ptr), isl.isl_id_copy(args[1].ptr), args[2])
            obj = space(ctx=ctx, ptr=res)
            return obj
        raise Error
    def add_unnamed_tuple(*args):
        if len(args) == 2 and type(args[1]) == int:
            ctx = args[0].ctx
            res = isl.isl_space_add_unnamed_tuple_ui(isl.isl_space_copy(args[0].ptr), args[1])
            obj = space(ctx=ctx, ptr=res)
            return obj
        raise Error
    def curry(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_curry(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def domain(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_domain(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def flatten_domain(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_flatten_domain(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def flatten_range(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_flatten_range(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_wrapping(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_is_wrapping(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def map_from_set(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_map_from_set(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def params(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_params(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is space:
                arg1 = space(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_product(isl.isl_space_copy(arg0.ptr), isl.isl_space_copy(arg1.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def range(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_range(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def range_reverse(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_range_reverse(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def reverse(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_reverse(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def uncurry(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_uncurry(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def unit():
        ctx = Context.getDefaultInstance()
        res = isl.isl_space_unit(ctx)
        obj = space(ctx=ctx, ptr=res)
        return obj
    def unwrap(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_unwrap(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj
    def wrap(arg0):
        try:
            if not arg0.__class__ is space:
                arg0 = space(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_space_wrap(isl.isl_space_copy(arg0.ptr))
        obj = space(ctx=ctx, ptr=res)
        return obj

isl.isl_space_add_named_tuple_id_ui.restype = c_void_p
isl.isl_space_add_named_tuple_id_ui.argtypes = [c_void_p, c_void_p, c_int]
isl.isl_space_add_unnamed_tuple_ui.restype = c_void_p
isl.isl_space_add_unnamed_tuple_ui.argtypes = [c_void_p, c_int]
isl.isl_space_curry.restype = c_void_p
isl.isl_space_curry.argtypes = [c_void_p]
isl.isl_space_domain.restype = c_void_p
isl.isl_space_domain.argtypes = [c_void_p]
isl.isl_space_flatten_domain.restype = c_void_p
isl.isl_space_flatten_domain.argtypes = [c_void_p]
isl.isl_space_flatten_range.restype = c_void_p
isl.isl_space_flatten_range.argtypes = [c_void_p]
isl.isl_space_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_space_is_wrapping.argtypes = [c_void_p]
isl.isl_space_map_from_set.restype = c_void_p
isl.isl_space_map_from_set.argtypes = [c_void_p]
isl.isl_space_params.restype = c_void_p
isl.isl_space_params.argtypes = [c_void_p]
isl.isl_space_product.restype = c_void_p
isl.isl_space_product.argtypes = [c_void_p, c_void_p]
isl.isl_space_range.restype = c_void_p
isl.isl_space_range.argtypes = [c_void_p]
isl.isl_space_range_reverse.restype = c_void_p
isl.isl_space_range_reverse.argtypes = [c_void_p]
isl.isl_space_reverse.restype = c_void_p
isl.isl_space_reverse.argtypes = [c_void_p]
isl.isl_space_uncurry.restype = c_void_p
isl.isl_space_uncurry.argtypes = [c_void_p]
isl.isl_space_unit.restype = c_void_p
isl.isl_space_unit.argtypes = [Context]
isl.isl_space_unwrap.restype = c_void_p
isl.isl_space_unwrap.argtypes = [c_void_p]
isl.isl_space_wrap.restype = c_void_p
isl.isl_space_wrap.argtypes = [c_void_p]
isl.isl_space_copy.restype = c_void_p
isl.isl_space_copy.argtypes = [c_void_p]
isl.isl_space_free.restype = c_void_p
isl.isl_space_free.argtypes = [c_void_p]
isl.isl_space_to_str.restype = POINTER(c_char)
isl.isl_space_to_str.argtypes = [c_void_p]

class union_access_info(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is union_map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_access_info_from_sink(isl.isl_union_map_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_access_info_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        ptr = isl.isl_union_access_info_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_access_info("""%s""")' % s
        else:
            return 'isl.union_access_info("%s")' % s
    def compute_flow(arg0):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_compute_flow(isl.isl_union_access_info_copy(arg0.ptr))
        obj = union_flow(ctx=ctx, ptr=res)
        return obj
    def set_kill(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_kill(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_access_info(ctx=ctx, ptr=res)
        return obj
    def set_may_source(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_may_source(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_access_info(ctx=ctx, ptr=res)
        return obj
    def set_must_source(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_must_source(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_access_info(ctx=ctx, ptr=res)
        return obj
    def set_schedule(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is schedule:
                arg1 = schedule(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_schedule(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_schedule_copy(arg1.ptr))
        obj = union_access_info(ctx=ctx, ptr=res)
        return obj
    def set_schedule_map(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_schedule_map(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        obj = union_access_info(ctx=ctx, ptr=res)
        return obj

isl.isl_union_access_info_from_sink.restype = c_void_p
isl.isl_union_access_info_from_sink.argtypes = [c_void_p]
isl.isl_union_access_info_compute_flow.restype = c_void_p
isl.isl_union_access_info_compute_flow.argtypes = [c_void_p]
isl.isl_union_access_info_set_kill.restype = c_void_p
isl.isl_union_access_info_set_kill.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_set_may_source.restype = c_void_p
isl.isl_union_access_info_set_may_source.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_set_must_source.restype = c_void_p
isl.isl_union_access_info_set_must_source.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_set_schedule.restype = c_void_p
isl.isl_union_access_info_set_schedule.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_set_schedule_map.restype = c_void_p
isl.isl_union_access_info_set_schedule_map.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_copy.restype = c_void_p
isl.isl_union_access_info_copy.argtypes = [c_void_p]
isl.isl_union_access_info_free.restype = c_void_p
isl.isl_union_access_info_free.argtypes = [c_void_p]
isl.isl_union_access_info_to_str.restype = POINTER(c_char)
isl.isl_union_access_info_to_str.argtypes = [c_void_p]

class union_flow(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_flow_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ptr = isl.isl_union_flow_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_flow("""%s""")' % s
        else:
            return 'isl.union_flow("%s")' % s
    def full_may_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_full_may_dependence(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_full_may_dependence(arg0):
        return arg0.full_may_dependence()
    def full_must_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_full_must_dependence(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_full_must_dependence(arg0):
        return arg0.full_must_dependence()
    def may_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_may_dependence(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_may_dependence(arg0):
        return arg0.may_dependence()
    def may_no_source(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_may_no_source(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_may_no_source(arg0):
        return arg0.may_no_source()
    def must_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_must_dependence(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_must_dependence(arg0):
        return arg0.must_dependence()
    def must_no_source(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_must_no_source(arg0.ptr)
        obj = union_map(ctx=ctx, ptr=res)
        return obj
    def get_must_no_source(arg0):
        return arg0.must_no_source()

isl.isl_union_flow_get_full_may_dependence.restype = c_void_p
isl.isl_union_flow_get_full_may_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_full_must_dependence.restype = c_void_p
isl.isl_union_flow_get_full_must_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_may_dependence.restype = c_void_p
isl.isl_union_flow_get_may_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_may_no_source.restype = c_void_p
isl.isl_union_flow_get_may_no_source.argtypes = [c_void_p]
isl.isl_union_flow_get_must_dependence.restype = c_void_p
isl.isl_union_flow_get_must_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_must_no_source.restype = c_void_p
isl.isl_union_flow_get_must_no_source.argtypes = [c_void_p]
isl.isl_union_flow_copy.restype = c_void_p
isl.isl_union_flow_copy.argtypes = [c_void_p]
isl.isl_union_flow_free.restype = c_void_p
isl.isl_union_flow_free.argtypes = [c_void_p]
isl.isl_union_flow_to_str.restype = POINTER(c_char)
isl.isl_union_flow_to_str.argtypes = [c_void_p]

class union_pw_aff_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_aff_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is union_pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_aff_list_from_union_pw_aff(isl.isl_union_pw_aff_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_pw_aff_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        ptr = isl.isl_union_pw_aff_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_pw_aff_list("""%s""")' % s
        else:
            return 'isl.union_pw_aff_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_aff:
                arg1 = union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_add(isl.isl_union_pw_aff_list_copy(arg0.ptr), isl.isl_union_pw_aff_copy(arg1.ptr))
        obj = union_pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_clear(isl.isl_union_pw_aff_list_copy(arg0.ptr))
        obj = union_pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_aff_list:
                arg1 = union_pw_aff_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_concat(isl.isl_union_pw_aff_list_copy(arg0.ptr), isl.isl_union_pw_aff_list_copy(arg1.ptr))
        obj = union_pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_drop(isl.isl_union_pw_aff_list_copy(arg0.ptr), arg1, arg2)
        obj = union_pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = union_pw_aff(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_get_at(arg0.ptr, arg1)
        obj = union_pw_aff(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is union_pw_aff:
                arg2 = union_pw_aff(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_insert(isl.isl_union_pw_aff_list_copy(arg0.ptr), arg1, isl.isl_union_pw_aff_copy(arg2.ptr))
        obj = union_pw_aff_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is union_pw_aff_list:
                arg0 = union_pw_aff_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_union_pw_aff_list_alloc.restype = c_void_p
isl.isl_union_pw_aff_list_alloc.argtypes = [Context, c_int]
isl.isl_union_pw_aff_list_from_union_pw_aff.restype = c_void_p
isl.isl_union_pw_aff_list_from_union_pw_aff.argtypes = [c_void_p]
isl.isl_union_pw_aff_list_add.restype = c_void_p
isl.isl_union_pw_aff_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_list_clear.restype = c_void_p
isl.isl_union_pw_aff_list_clear.argtypes = [c_void_p]
isl.isl_union_pw_aff_list_concat.restype = c_void_p
isl.isl_union_pw_aff_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_list_drop.restype = c_void_p
isl.isl_union_pw_aff_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_union_pw_aff_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_pw_aff_list_get_at.restype = c_void_p
isl.isl_union_pw_aff_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_union_pw_aff_list_insert.restype = c_void_p
isl.isl_union_pw_aff_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_union_pw_aff_list_size.argtypes = [c_void_p]
isl.isl_union_pw_aff_list_copy.restype = c_void_p
isl.isl_union_pw_aff_list_copy.argtypes = [c_void_p]
isl.isl_union_pw_aff_list_free.restype = c_void_p
isl.isl_union_pw_aff_list_free.argtypes = [c_void_p]
isl.isl_union_pw_aff_list_to_str.restype = POINTER(c_char)
isl.isl_union_pw_aff_list_to_str.argtypes = [c_void_p]

class union_set_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is union_set:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_list_from_union_set(isl.isl_union_set_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_set_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        ptr = isl.isl_union_set_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_set_list("""%s""")' % s
        else:
            return 'isl.union_set_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_list_add(isl.isl_union_set_list_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        obj = union_set_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_list_clear(isl.isl_union_set_list_copy(arg0.ptr))
        obj = union_set_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set_list:
                arg1 = union_set_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_list_concat(isl.isl_union_set_list_copy(arg0.ptr), isl.isl_union_set_list_copy(arg1.ptr))
        obj = union_set_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_list_drop(isl.isl_union_set_list_copy(arg0.ptr), arg1, arg2)
        obj = union_set_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = union_set(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_set_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_list_get_at(arg0.ptr, arg1)
        obj = union_set(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is union_set:
                arg2 = union_set(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_list_insert(isl.isl_union_set_list_copy(arg0.ptr), arg1, isl.isl_union_set_copy(arg2.ptr))
        obj = union_set_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is union_set_list:
                arg0 = union_set_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_union_set_list_alloc.restype = c_void_p
isl.isl_union_set_list_alloc.argtypes = [Context, c_int]
isl.isl_union_set_list_from_union_set.restype = c_void_p
isl.isl_union_set_list_from_union_set.argtypes = [c_void_p]
isl.isl_union_set_list_add.restype = c_void_p
isl.isl_union_set_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_list_clear.restype = c_void_p
isl.isl_union_set_list_clear.argtypes = [c_void_p]
isl.isl_union_set_list_concat.restype = c_void_p
isl.isl_union_set_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_list_drop.restype = c_void_p
isl.isl_union_set_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_union_set_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_set_list_get_at.restype = c_void_p
isl.isl_union_set_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_union_set_list_insert.restype = c_void_p
isl.isl_union_set_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_union_set_list_size.argtypes = [c_void_p]
isl.isl_union_set_list_copy.restype = c_void_p
isl.isl_union_set_list_copy.argtypes = [c_void_p]
isl.isl_union_set_list_free.restype = c_void_p
isl.isl_union_set_list_free.argtypes = [c_void_p]
isl.isl_union_set_list_to_str.restype = POINTER(c_char)
isl.isl_union_set_list_to_str.argtypes = [c_void_p]

class val(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_val_int_from_si(self.ctx, args[0])
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_val_read_from_str(self.ctx, args[0].encode('ascii'))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_val_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ptr = isl.isl_val_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.val("""%s""")' % s
        else:
            return 'isl.val("%s")' % s
    def abs(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_abs(isl.isl_val_copy(arg0.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def abs_eq(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_abs_eq(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_add(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def ceil(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_ceil(isl.isl_val_copy(arg0.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def cmp_si(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_cmp_si(arg0.ptr, arg1)
        return res
    def div(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_div(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def eq(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_eq(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def floor(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_floor(isl.isl_val_copy(arg0.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def gcd(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_gcd(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def ge(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_ge(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def den_si(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_get_den_si(arg0.ptr)
        return res
    def get_den_si(arg0):
        return arg0.den_si()
    def num_si(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_get_num_si(arg0.ptr)
        return res
    def get_num_si(arg0):
        return arg0.num_si()
    def gt(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_gt(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    @staticmethod
    def infty():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_infty(ctx)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def inv(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_inv(isl.isl_val_copy(arg0.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def is_divisible_by(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_divisible_by(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_infty(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_infty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_int(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_int(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_nan(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_nan(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_neg(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_neg(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_neginfty(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_neginfty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_negone(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_negone(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_nonneg(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_nonneg(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_nonpos(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_nonpos(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_one(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_one(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_pos(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_pos(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_rat(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_rat(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_zero(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_zero(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def le(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_le(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lt(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_lt(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def max(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_max(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def min(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_min(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def mod(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_mod(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def mul(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_mul(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def nan():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_nan(ctx)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def ne(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_ne(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def neg(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_neg(isl.isl_val_copy(arg0.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def neginfty():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_neginfty(ctx)
        obj = val(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def negone():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_negone(ctx)
        obj = val(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def one():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_one(ctx)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def pow2(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_pow2(isl.isl_val_copy(arg0.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def sgn(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_sgn(arg0.ptr)
        return res
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_sub(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    def trunc(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_trunc(isl.isl_val_copy(arg0.ptr))
        obj = val(ctx=ctx, ptr=res)
        return obj
    @staticmethod
    def zero():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_zero(ctx)
        obj = val(ctx=ctx, ptr=res)
        return obj

isl.isl_val_int_from_si.restype = c_void_p
isl.isl_val_int_from_si.argtypes = [Context, c_long]
isl.isl_val_read_from_str.restype = c_void_p
isl.isl_val_read_from_str.argtypes = [Context, c_char_p]
isl.isl_val_abs.restype = c_void_p
isl.isl_val_abs.argtypes = [c_void_p]
isl.isl_val_abs_eq.argtypes = [c_void_p, c_void_p]
isl.isl_val_add.restype = c_void_p
isl.isl_val_add.argtypes = [c_void_p, c_void_p]
isl.isl_val_ceil.restype = c_void_p
isl.isl_val_ceil.argtypes = [c_void_p]
isl.isl_val_cmp_si.argtypes = [c_void_p, c_long]
isl.isl_val_div.restype = c_void_p
isl.isl_val_div.argtypes = [c_void_p, c_void_p]
isl.isl_val_eq.argtypes = [c_void_p, c_void_p]
isl.isl_val_floor.restype = c_void_p
isl.isl_val_floor.argtypes = [c_void_p]
isl.isl_val_gcd.restype = c_void_p
isl.isl_val_gcd.argtypes = [c_void_p, c_void_p]
isl.isl_val_ge.argtypes = [c_void_p, c_void_p]
isl.isl_val_get_den_si.argtypes = [c_void_p]
isl.isl_val_get_num_si.argtypes = [c_void_p]
isl.isl_val_gt.argtypes = [c_void_p, c_void_p]
isl.isl_val_infty.restype = c_void_p
isl.isl_val_infty.argtypes = [Context]
isl.isl_val_inv.restype = c_void_p
isl.isl_val_inv.argtypes = [c_void_p]
isl.isl_val_is_divisible_by.argtypes = [c_void_p, c_void_p]
isl.isl_val_is_infty.argtypes = [c_void_p]
isl.isl_val_is_int.argtypes = [c_void_p]
isl.isl_val_is_nan.argtypes = [c_void_p]
isl.isl_val_is_neg.argtypes = [c_void_p]
isl.isl_val_is_neginfty.argtypes = [c_void_p]
isl.isl_val_is_negone.argtypes = [c_void_p]
isl.isl_val_is_nonneg.argtypes = [c_void_p]
isl.isl_val_is_nonpos.argtypes = [c_void_p]
isl.isl_val_is_one.argtypes = [c_void_p]
isl.isl_val_is_pos.argtypes = [c_void_p]
isl.isl_val_is_rat.argtypes = [c_void_p]
isl.isl_val_is_zero.argtypes = [c_void_p]
isl.isl_val_le.argtypes = [c_void_p, c_void_p]
isl.isl_val_lt.argtypes = [c_void_p, c_void_p]
isl.isl_val_max.restype = c_void_p
isl.isl_val_max.argtypes = [c_void_p, c_void_p]
isl.isl_val_min.restype = c_void_p
isl.isl_val_min.argtypes = [c_void_p, c_void_p]
isl.isl_val_mod.restype = c_void_p
isl.isl_val_mod.argtypes = [c_void_p, c_void_p]
isl.isl_val_mul.restype = c_void_p
isl.isl_val_mul.argtypes = [c_void_p, c_void_p]
isl.isl_val_nan.restype = c_void_p
isl.isl_val_nan.argtypes = [Context]
isl.isl_val_ne.argtypes = [c_void_p, c_void_p]
isl.isl_val_neg.restype = c_void_p
isl.isl_val_neg.argtypes = [c_void_p]
isl.isl_val_neginfty.restype = c_void_p
isl.isl_val_neginfty.argtypes = [Context]
isl.isl_val_negone.restype = c_void_p
isl.isl_val_negone.argtypes = [Context]
isl.isl_val_one.restype = c_void_p
isl.isl_val_one.argtypes = [Context]
isl.isl_val_pow2.restype = c_void_p
isl.isl_val_pow2.argtypes = [c_void_p]
isl.isl_val_sgn.argtypes = [c_void_p]
isl.isl_val_sub.restype = c_void_p
isl.isl_val_sub.argtypes = [c_void_p, c_void_p]
isl.isl_val_trunc.restype = c_void_p
isl.isl_val_trunc.argtypes = [c_void_p]
isl.isl_val_zero.restype = c_void_p
isl.isl_val_zero.argtypes = [Context]
isl.isl_val_copy.restype = c_void_p
isl.isl_val_copy.argtypes = [c_void_p]
isl.isl_val_free.restype = c_void_p
isl.isl_val_free.argtypes = [c_void_p]
isl.isl_val_to_str.restype = POINTER(c_char)
isl.isl_val_to_str.argtypes = [c_void_p]

class val_list(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_val_list_alloc(self.ctx, args[0])
            return
        if len(args) == 1 and (args[0].__class__ is val or type(args[0]) == int):
            args = list(args)
            try:
                if not args[0].__class__ is val:
                    args[0] = val(args[0])
            except:
                raise
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_val_list_from_val(isl.isl_val_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_val_list_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        ptr = isl.isl_val_list_to_str(arg0.ptr)
        res = cast(ptr, c_char_p).value.decode('ascii')
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.val_list("""%s""")' % s
        else:
            return 'isl.val_list("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_list_add(isl.isl_val_list_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        obj = val_list(ctx=ctx, ptr=res)
        return obj
    def clear(arg0):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_list_clear(isl.isl_val_list_copy(arg0.ptr))
        obj = val_list(ctx=ctx, ptr=res)
        return obj
    def concat(arg0, arg1):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val_list:
                arg1 = val_list(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_list_concat(isl.isl_val_list_copy(arg0.ptr), isl.isl_val_list_copy(arg1.ptr))
        obj = val_list(ctx=ctx, ptr=res)
        return obj
    def drop(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_list_drop(isl.isl_val_list_copy(arg0.ptr), arg1, arg2)
        obj = val_list(ctx=ctx, ptr=res)
        return obj
    def foreach(arg0, arg1):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = val(ctx=arg0.ctx, ptr=(cb_arg0))
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_val_list_foreach(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        if res < 0:
            raise
    def at(arg0, arg1):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_list_get_at(arg0.ptr, arg1)
        obj = val(ctx=ctx, ptr=res)
        return obj
    def get_at(arg0, arg1):
        return arg0.at(arg1)
    def insert(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        try:
            if not arg2.__class__ is val:
                arg2 = val(arg2)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_list_insert(isl.isl_val_list_copy(arg0.ptr), arg1, isl.isl_val_copy(arg2.ptr))
        obj = val_list(ctx=ctx, ptr=res)
        return obj
    def size(arg0):
        try:
            if not arg0.__class__ is val_list:
                arg0 = val_list(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_list_size(arg0.ptr)
        if res < 0:
            raise
        return int(res)

isl.isl_val_list_alloc.restype = c_void_p
isl.isl_val_list_alloc.argtypes = [Context, c_int]
isl.isl_val_list_from_val.restype = c_void_p
isl.isl_val_list_from_val.argtypes = [c_void_p]
isl.isl_val_list_add.restype = c_void_p
isl.isl_val_list_add.argtypes = [c_void_p, c_void_p]
isl.isl_val_list_clear.restype = c_void_p
isl.isl_val_list_clear.argtypes = [c_void_p]
isl.isl_val_list_concat.restype = c_void_p
isl.isl_val_list_concat.argtypes = [c_void_p, c_void_p]
isl.isl_val_list_drop.restype = c_void_p
isl.isl_val_list_drop.argtypes = [c_void_p, c_int, c_int]
isl.isl_val_list_foreach.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_val_list_get_at.restype = c_void_p
isl.isl_val_list_get_at.argtypes = [c_void_p, c_int]
isl.isl_val_list_insert.restype = c_void_p
isl.isl_val_list_insert.argtypes = [c_void_p, c_int, c_void_p]
isl.isl_val_list_size.argtypes = [c_void_p]
isl.isl_val_list_copy.restype = c_void_p
isl.isl_val_list_copy.argtypes = [c_void_p]
isl.isl_val_list_free.restype = c_void_p
isl.isl_val_list_free.argtypes = [c_void_p]
isl.isl_val_list_to_str.restype = POINTER(c_char)
isl.isl_val_list_to_str.argtypes = [c_void_p]
