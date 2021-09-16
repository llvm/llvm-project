#!/usr/bin/env python

import sys


class CType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class GMPAPI:
    def __init__(self, ret_ty, name, *params, **kw):
        out = kw.get('out', [0])
        inout = kw.get('inout', [])
        mixed = kw.get('mixed', False)
        custom = kw.get('custom', False)
        self.name = name
        self.ret_ty = ret_ty
        self.params = params
        self.inout_params = inout
        self.custom_test = custom
        # most functions with return results dont need extra out params
        # set mixed to true to check both the return value and an out param
        if self.ret_ty != void and not mixed:
            self.out_params = []
        else:
            self.out_params = out  #param location of the output result

    def is_write_only(self, pos):
        if pos in self.out_params and pos not in self.inout_params:
            return True
        return False

    def __str__(self):
        return ("{} {}({})".format(self.ret_ty, self.name, ",".join(
            map(str, self.params))))

    def __repr__(self):
        return str(self)


void = CType("void")
voidp = CType("void *")
charp = CType("char *")
iint = CType("int")
size_t = CType("size_t")
size_tp = CType("size_t*")
ilong = CType("long")
ulong = CType("unsigned long")
mpz_t = CType("mpz_t")
mpq_t = CType("mpq_t")

apis = [
    GMPAPI(void, "mpz_abs", mpz_t, mpz_t),
    GMPAPI(void, "mpz_add", mpz_t, mpz_t, mpz_t),
    GMPAPI(iint, "mpz_cmp_si", mpz_t, ilong),
    GMPAPI(iint, "mpz_cmpabs", mpz_t, mpz_t),
    GMPAPI(iint, "mpz_cmp", mpz_t, mpz_t),
    GMPAPI(void, "mpz_mul", mpz_t, mpz_t, mpz_t),
    GMPAPI(void, "mpz_neg", mpz_t, mpz_t),
    GMPAPI(void, "mpz_set_si", mpz_t, ilong),
    GMPAPI(void, "mpz_set", mpz_t, mpz_t),
    GMPAPI(void, "mpz_sub", mpz_t, mpz_t, mpz_t),
    GMPAPI(void, "mpz_swap", mpz_t, mpz_t, out=[0, 1], inout=[0, 1]),
    GMPAPI(iint, "mpz_sgn", mpz_t),
    GMPAPI(void, "mpz_addmul", mpz_t, mpz_t, mpz_t, inout=[0]),
    GMPAPI(void, "mpz_divexact", mpz_t, mpz_t, mpz_t),
    GMPAPI(iint, "mpz_divisible_p", mpz_t, mpz_t),
    GMPAPI(void, "mpz_submul", mpz_t, mpz_t, mpz_t, inout=[0]),
    GMPAPI(void, "mpz_set_ui", mpz_t, ulong),
    GMPAPI(void, "mpz_add_ui", mpz_t, mpz_t, ulong),
    GMPAPI(void, "mpz_divexact_ui", mpz_t, mpz_t, ulong),
    GMPAPI(void, "mpz_mul_ui", mpz_t, mpz_t, ulong),
    GMPAPI(void, "mpz_pow_ui", mpz_t, mpz_t, ulong),
    GMPAPI(void, "mpz_sub_ui", mpz_t, mpz_t, ulong),
    GMPAPI(void, "mpz_cdiv_q", mpz_t, mpz_t, mpz_t),
    GMPAPI(void, "mpz_fdiv_q", mpz_t, mpz_t, mpz_t),
    GMPAPI(void, "mpz_fdiv_r", mpz_t, mpz_t, mpz_t),
    GMPAPI(void, "mpz_tdiv_q", mpz_t, mpz_t, mpz_t),
    GMPAPI(ulong, "mpz_fdiv_q_ui", mpz_t, mpz_t, ulong, out=[0], mixed=True),
    GMPAPI(ilong, "mpz_get_si", mpz_t),
    GMPAPI(ulong, "mpz_get_ui", mpz_t),
    GMPAPI(void, "mpz_gcd", mpz_t, mpz_t, mpz_t),
    GMPAPI(void, "mpz_lcm", mpz_t, mpz_t, mpz_t),
    GMPAPI(void, "mpz_mul_2exp", mpz_t, mpz_t, ulong),
    GMPAPI(
        void,
        "mpz_export",
        voidp,
        size_tp,
        iint,
        size_t,
        iint,
        size_t,
        mpz_t,
        custom=True),
    # The mpz_import signature is a bit of a lie, but it is ok because it is custom
    GMPAPI(
        void,
        "mpz_import",
        voidp,
        size_t,
        iint,
        size_t,
        iint,
        size_t,
        mpz_t,
        custom=True),
    GMPAPI(size_t, "mpz_sizeinbase", mpz_t, iint),
    GMPAPI(charp, "mpz_get_str", charp, iint, mpz_t),

    # mpq functions
    GMPAPI(iint, "mpq_set_str", mpq_t, charp, iint, out=[0], mixed=True),
    GMPAPI(void, "mpq_canonicalize", mpq_t, inout=[0]),
    GMPAPI(iint, "mpq_cmp", mpq_t, mpq_t),
    GMPAPI(void, "mpq_mul", mpq_t, mpq_t, mpq_t),
    GMPAPI(void, "mpq_set", mpq_t, mpq_t),
    GMPAPI(void, "mpq_set_ui", mpq_t, ulong, ulong),
    GMPAPI(iint, "mpq_sgn", mpq_t),
    GMPAPI(charp, "mpq_get_str", charp, iint, mpq_t),
]


def get_api(name):
    for a in apis:
        if a.name == name:
            return a
    raise RuntimeError("Unknown api: {}".format(name))
