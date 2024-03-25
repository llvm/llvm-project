#!/usr/bin/env python
import random
import gmpapi

MAX_SLONG = "9223372036854775807"
MIN_SLONG = "-9223372036854775808"
MAX_ULONG = "18446744073709551615"
MAX_SINT = "2147483647"
MIN_SINT = "-2147483648"
MAX_UINT = "4294967295"
MAX_SSHORT = "32767"
MIN_SSHORT = "-32768"
MAX_USHORT = "65535"


def plus1(x):
    return str(int(x) + 1)


def minus1(x):
    return str(int(x) - 1)


def apply(fun, lst):
    return list(map(str, map(fun, lst)))


mzero_one = ["-0", "-1"]
zero_one = ["0", "1"]
mm_slong = [MAX_SLONG, MIN_SLONG]
mm_slong1 = [minus1(MAX_SLONG), plus1(MIN_SLONG)]
mm_ulong = [MAX_ULONG]
mm_ulong1 = [minus1(MAX_ULONG)]
mm_sint = [MAX_SINT, MIN_SINT]
mm_sint1 = [minus1(MAX_SINT), plus1(MIN_SINT)]
mm_uint = [MAX_UINT]
mm_uint1 = [minus1(MAX_UINT)]
mm_sshort = [MAX_SSHORT, MIN_SSHORT]
mm_sshort1 = [minus1(MAX_SSHORT), plus1(MIN_SSHORT)]
mm_ushort = [MAX_USHORT]
mm_ushort1 = [minus1(MAX_USHORT)]
mm_all = mm_slong + mm_ulong + mm_sint + mm_uint + mm_sshort + mm_ushort
zero_one_all = mzero_one + zero_one

mpz_std_list = zero_one_all + mm_all + apply(plus1, mm_all) + apply(minus1, mm_all)
si_std_list = (
    zero_one + mm_slong + mm_sint + mm_sshort + mm_slong1 + mm_sint1 + mm_sshort1
)
ui_std_list = (
    zero_one + mm_ulong + mm_uint + mm_ushort + mm_ulong1 + mm_uint1 + mm_ushort1
)


def gen_random_mpz(mindigits=1, maxdigits=100, allowneg=True):
    sign = random.choice(["", "-"])
    if not allowneg:
        sign = ""
    return sign + gen_digits(random.randint(mindigits, maxdigits))


def gen_random_si():
    si = gen_random_mpz(mindigits=1, maxdigits=19)
    while int(si) > int(MAX_SLONG) or int(si) < int(MIN_SLONG):
        si = gen_random_mpz(mindigits=1, maxdigits=19)
    return si


def gen_random_ui():
    ui = gen_random_mpz(mindigits=1, maxdigits=20, allowneg=False)
    while int(ui) > int(MAX_ULONG):
        ui = gen_random_mpz(mindigits=1, maxdigits=20, allowneg=False)
    return ui


def gen_digits(length):
    if length == 1:
        i = random.randint(1, 9)
    else:
        digits = [random.randint(1, 9)] + [
            random.randint(0, 9) for x in range(length - 1)
        ]
        digits = map(str, digits)
        i = "".join(digits)
    return str(i)


def gen_mpzs(mindigits=1, maxdigits=100, count=10):
    return [
        gen_random_mpz(mindigits=mindigits, maxdigits=maxdigits) for x in range(count)
    ]


default_count = 10


def gen_sis(count=default_count):
    return [gen_random_si() for x in range(count)]


def gen_uis(count=default_count):
    return [gen_random_ui() for x in range(count)]


def gen_small_mpzs(count=default_count):
    return gen_mpzs(mindigits=1, maxdigits=4, count=count)


def is_small_mpz(s):
    return len(s) >= 1 and len(s) <= 4


def gen_medium_mpzs(count=default_count):
    return gen_mpzs(mindigits=5, maxdigits=20, count=count)


def is_medium_mpz(s):
    return len(s) >= 5 and len(s) <= 20


def gen_large_mpzs(count=default_count):
    return gen_mpzs(mindigits=21, maxdigits=100, count=count)


def is_large_mpz(s):
    return len(s) >= 21


def gen_mpz_spread(count=default_count):
    return gen_small_mpzs(count) + gen_medium_mpzs(count) + gen_large_mpzs(count)


def gen_mpz_args(count=default_count):
    return mpz_std_list + gen_mpz_spread(count)


def gen_mpq_args(count=4):
    nums = zero_one + gen_mpz_spread(count)
    dens = ["1"] + gen_mpz_spread(count)
    return [n + "/" + d for n in nums for d in dens if int(d) != 0]


def gen_si_args():
    return si_std_list + gen_sis()


def gen_ui_args():
    return ui_std_list + gen_uis()


def gen_list_for_type(t, is_write_only):
    if (t == gmpapi.mpz_t or t == gmpapi.mpq_t) and is_write_only:
        return ["0"]
    elif t == gmpapi.mpz_t:
        return gen_mpz_args()
    elif t == gmpapi.ilong:
        return gen_si_args()
    elif t == gmpapi.ulong:
        return gen_ui_args()
    elif t == gmpapi.mpq_t:
        return gen_mpq_args()
    else:
        raise RuntimeError("Unknown type: {}".format(t))


def gen_args(api):
    if api.custom_test or api.name in custom:
        return custom[api.name](api)
    types = api.params
    if len(types) == 1:
        return [[a] for a in gen_list_for_type(types[0], api.is_write_only(0))]
    elif len(types) == 2:
        t1 = gen_list_for_type(types[0], api.is_write_only(0))
        t2 = gen_list_for_type(types[1], api.is_write_only(1))
        return [(a, b) for a in t1 for b in t2]
    elif len(types) == 3:
        t1 = gen_list_for_type(types[0], api.is_write_only(0))
        t2 = gen_list_for_type(types[1], api.is_write_only(1))
        t3 = gen_list_for_type(types[2], api.is_write_only(2))
        return [(a, b, c) for a in t1 for b in t2 for c in t3]
    elif len(types) == 4:
        t1 = gen_list_for_type(types[0], api.is_write_only(0))
        t2 = gen_list_for_type(types[1], api.is_write_only(1))
        t3 = gen_list_for_type(types[2], api.is_write_only(2))
        t4 = gen_list_for_type(types[3], api.is_write_only(3))
        return [(a, b, c, d) for a in t1 for b in t2 for c in t3 for d in t4]
    else:
        raise RuntimeError("Too many args: {}".format(len(types)))


###################################################################
#
# Fixup and massage random data for better test coverage
#
###################################################################
def mul_mpzs(a, b):
    return str(int(a) * int(b))


def mpz_divexact_data(args):
    # set n = n * d
    divisible = mul_mpzs(args[1], (args[2]))
    return [(args[0], divisible, args[2])]


def mpz_divisible_p_data(args):
    (n, d) = get_div_data(args[0], args[1], rate=1.0)
    return [(n, d), (args[0], args[1])]


def mpz_div3_data(args):
    q = args[0]
    (n, d) = get_div_data(args[1], args[2], rate=1.0)
    return [(q, n, d), (q, args[1], args[2])]


def mpz_pow_data(args, alwaysallowbase1=True):
    base = int(args[1])
    exp = int(args[2])
    # allow special numbers
    if base == 0 or exp == 0 or exp == 1:
        return [args]
    if base == 1 and alwaysallowbase1:
        return [args]

    # disallow too big numbers
    if base > 1000 or base < -1000:
        base = gen_random_mpz(maxdigits=3)
    if exp > 1000:
        exp = gen_random_mpz(maxdigits=3, allowneg=False)

    return [(args[0], str(base), str(exp))]


def mpz_mul_2exp_data(args):
    return mpz_pow_data(args, alwaysallowbase1=False)


def mpz_gcd_data(args):
    r = args[0]
    a = args[1]
    b = args[2]
    s_ = gen_small_mpzs(1)[0]
    m_ = gen_medium_mpzs(1)[0]
    l_ = gen_large_mpzs(1)[0]

    return [
        (r, a, b),
        (r, mul_mpzs(a, b), b),
        (r, mul_mpzs(a, s_), mul_mpzs(b, s_)),
        (r, mul_mpzs(a, m_), mul_mpzs(b, m_)),
        (r, mul_mpzs(a, l_), mul_mpzs(b, l_)),
    ]


def mpz_export_data(api):
    rop = ["0"]
    countp = ["0"]
    order = ["-1", "1"]
    size = ["1", "2", "4", "8"]
    endian = ["0"]
    nails = ["0"]
    ops = gen_mpz_args(1000) + gen_mpzs(count=100, mindigits=100, maxdigits=1000)

    args = []
    for r in rop:
        for c in countp:
            for o in order:
                for s in size:
                    for e in endian:
                        for n in nails:
                            for op in ops:
                                args.append((r, c, o, s, e, n, op))
    return args


def mpz_sizeinbase_data(api):
    bases = list(map(str, range(2, 37)))
    ops = gen_mpz_args(1000) + gen_mpzs(count=1000, mindigits=100, maxdigits=2000)
    return [(op, b) for op in ops for b in bases]


def get_str_data(ty):
    bases = list(range(2, 37)) + list(range(-2, -37, -1))
    bases = list(map(str, bases))
    if ty == gmpapi.mpz_t:
        ops = gen_mpz_args(1000)
    elif ty == gmpapi.mpq_t:
        ops = gen_mpq_args(20)
    else:
        raise RuntimeError("Unsupported get_str type: " + str(ty))
    return [("NULL", b, op) for b in bases for op in ops]


def mpz_get_str_data(api):
    return get_str_data(gmpapi.mpz_t)


def mpq_get_str_data(api):
    return get_str_data(gmpapi.mpq_t)


def mpq_set_str_data(api):
    args = gen_mpq_args(20) + gen_mpz_args()
    # zero does not match results exactly because the
    # results are not canonicalized first. We choose to
    # exclude zero from test results. The other option is
    # to canonicalize the results after parsing the strings.
    # Instead we exclude zero so that we can independently
    # test correctness of set_str and canonicalization
    nonzero = []
    for arg in args:
        if "/" in arg:
            pos = arg.find("/")
            if int(arg[:pos]) != 0:
                nonzero.append(arg)
        elif int(arg) != 0:
            nonzero.append(arg)

    return [("0", q, "10") for q in nonzero]


def get_div_data(n, d, rate=0.2):
    """Generate some inputs that are perfectly divisible"""
    if random.random() < rate:
        n = mul_mpzs(n, d)
    return (n, d)


def allow(name, args):
    if name not in blacklists:
        return True
    filters = blacklists[name]
    for (pos, disallow) in filters:
        if args[pos] in disallow:
            return False
    return True


def fixup_args(name, args):
    if name not in fixups:
        return [args]
    return fixups[name](args)


# list of values to be excluded for various api calls
# list format is (pos, [list of values to exclude])
blacklists = {
    "mpz_cdiv_q": [(2, ["0", "-0"])],
    "mpz_fdiv_q": [(2, ["0", "-0"])],
    "mpz_fdiv_r": [(2, ["0", "-0"])],
    "mpz_tdiv_q": [(2, ["0", "-0"])],
    "mpz_fdiv_q_ui": [(2, ["0", "-0"])],
    "mpz_divexact": [(2, ["0", "-0"])],
    "mpz_divisible_p": [(1, ["0", "-0"])],
    "mpz_divexact_ui": [(2, ["0", "-0"])],
    "mpq_set_ui": [(2, ["0", "-0"])],
}

fixups = {
    "mpz_divexact": mpz_divexact_data,
    "mpz_divisible_p": mpz_divisible_p_data,
    "mpz_cdiv_q": mpz_div3_data,
    "mpz_fdiv_q": mpz_div3_data,
    "mpz_fdiv_r": mpz_div3_data,
    "mpz_tdiv_q": mpz_div3_data,
    "mpz_fdiv_q_ui": mpz_div3_data,
    "mpz_divexact_ui": mpz_divexact_data,
    "mpz_pow_ui": mpz_pow_data,
    "mpz_gcd": mpz_gcd_data,
    "mpz_lcm": mpz_gcd_data,
    "mpz_mul_2exp": mpz_mul_2exp_data,
}

custom = {
    "mpz_export": mpz_export_data,
    "mpz_import": mpz_export_data,
    "mpz_sizeinbase": mpz_sizeinbase_data,
    "mpz_get_str": mpz_get_str_data,
    "mpq_set_str": mpq_set_str_data,
    "mpq_get_str": mpq_get_str_data,
}

if __name__ == "__main__":
    # apis = [gmpapi.get_api("mpq_set_str"),]
    apis = gmpapi.apis
    for api in apis:
        tests = gen_args(api)
        for args in tests:
            expanded_args = fixup_args(api.name, args)
            for args in expanded_args:
                if allow(api.name, args):
                    print("{}|{}".format(api.name, ",".join(args)))
