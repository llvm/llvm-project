#!/usr/bin/env python
import sys
import gmpapi
from gmpapi import void
from gmpapi import ilong
from gmpapi import iint
from gmpapi import ulong
from gmpapi import mpz_t
from gmpapi import size_t
from gmpapi import charp
from gmpapi import mpq_t


class APITest(object):
    def __init__(self, gmpapi):
        self.api = gmpapi

    def test_prefix(self):
        return "test"

    def test_param_name(self, ty, i):
        if ty == mpz_t:
            pname = "p_zs"
        elif ty == ilong:
            pname = "p_si"
        elif ty == ulong:
            pname = "p_ui"
        elif ty == iint:
            pname = "p_i"
        elif ty == charp:
            pname = "p_cs"
        elif ty == mpq_t:
            pname = "p_qs"
        else:
            raise RuntimeError("Unknown param type: " + str(ty))
        return pname + str(i)

    def test_param_type(self, ty):
        if ty == mpz_t or ty == mpq_t:
            pty_name = "char *"
        else:
            pty_name = str(ty)
        return pty_name

    def test_var_name(self, ty, i):
        if ty == mpz_t:
            vname = "v_z"
        elif ty == ilong:
            vname = "v_si"
        elif ty == ulong:
            vname = "v_ui"
        elif ty == iint:
            vname = "v_i"
        elif ty == size_t:
            vname = "v_st"
        elif ty == charp:
            vname = "v_cs"
        elif ty == mpq_t:
            vname = "v_q"
        else:
            raise RuntimeError("Unknown param type: " + str(ty))
        return vname + str(i)

    def test_var_type(self, ty):
        if ty == mpz_t:
            return self.mpz_type()
        elif ty == mpq_t:
            return self.mpq_type()
        else:
            return str(ty)

    def init_var_from_param(self, ty, var, param):
        code = "\t"
        if ty == mpz_t or ty == mpq_t:
            code += self.api_call_prefix(ty) + "init(" + var + ");\n\t"
            code += (
                self.api_call_prefix(ty)
                + "set_str("
                + ",".join([var, param, "10"])
                + ")"
            )
            if ty == mpq_t:
                code += ";\n\t"
                code += self.api_call_prefix(ty) + "canonicalize(" + var + ")"
        else:
            code += var + "=" + param
        return code

    def init_vars_from_params(self):
        code = ""
        for (i, p) in enumerate(self.api.params):
            param = self.test_param_name(p, i)
            code += "\t"
            code += self.test_var_type(p) + " "
            var = self.test_var_name(p, i)
            code += var + ";\n"
            code += self.init_var_from_param(p, var, param) + ";\n\n"
        return code

    def make_api_call(self):
        bare_name = self.api.name.replace("mpz_", "", 1).replace("mpq_", "", 1)
        call_params = [
            self.test_var_name(p, i) for (i, p) in enumerate(self.api.params)
        ]
        ret = "\t"
        ret_ty = self.api.ret_ty
        if ret_ty != void:
            ret += (
                self.test_var_type(ret_ty)
                + " "
                + self.test_var_name(ret_ty, "_ret")
                + " = "
            )
        # call mpq or mpz function
        if self.api.name.startswith("mpz_"):
            prefix = self.api_call_prefix(mpz_t)
        else:
            prefix = self.api_call_prefix(mpq_t)
        return ret + prefix + bare_name + "(" + ",".join(call_params) + ");\n"

    def normalize_cmp(self, ty):
        cmpval = self.test_var_name(ty, "_ret")
        code = ""
        code += """
	if ({var} > 0)
	  {var} = 1;
	else if ({var} < 0)
	  {var} = -1;\n\t
""".format(
            var=cmpval
        )
        return code

    def extract_result(self, ty, pos):
        code = ""
        if ty == mpz_t or ty == mpq_t:
            var = self.test_var_name(ty, pos)
            code += self.api_call_prefix(ty) + "get_str(out+offset, 10," + var + ");\n"
            code += "\toffset = offset + strlen(out); "
            code += "out[offset] = ' '; out[offset+1] = 0; offset += 1;"
        else:
            assert pos == -1, "expected a return value, not a param value"
            if ty == ilong:
                var = self.test_var_name(ty, "_ret")
                code += 'offset = sprintf(out+offset, " %ld ", ' + var + ");"
            elif ty == ulong:
                var = self.test_var_name(ty, "_ret")
                code += 'offset = sprintf(out+offset, " %lu ", ' + var + ");"
            elif ty == iint:
                var = self.test_var_name(ty, "_ret")
                code += 'offset = sprintf(out+offset, " %d ", ' + var + ");"
            elif ty == size_t:
                var = self.test_var_name(ty, "_ret")
                code += 'offset = sprintf(out+offset, " %zu ", ' + var + ");"
            elif ty == charp:
                var = self.test_var_name(ty, "_ret")
                code += 'offset = sprintf(out+offset, " %s ", ' + var + ");"
            else:
                raise RuntimeError("Unknown param type: " + str(ty))
        return code

    def extract_results(self):
        ret_ty = self.api.ret_ty
        code = "\tint offset = 0;\n\t"

        # normalize cmp return values
        if ret_ty == iint and "cmp" in self.api.name:
            code += self.normalize_cmp(ret_ty)

        # call canonicalize for mpq_set_ui
        if self.api.name == "mpq_set_ui":
            code += (
                self.api_call_prefix(mpq_t)
                + "canonicalize("
                + self.test_var_name(mpq_t, 0)
                + ");\n\t"
            )

        # get return value
        if ret_ty != void:
            code += self.extract_result(ret_ty, -1) + "\n"

        # get out param values
        for pos in self.api.out_params:
            code += "\t"
            code += self.extract_result(self.api.params[pos], pos) + "\n"

        return code + "\n"

    def clear_local_vars(self):
        code = ""
        for (i, p) in enumerate(self.api.params):
            if p == mpz_t or p == mpq_t:
                var = self.test_var_name(p, i)
                code += "\t" + self.api_call_prefix(p) + "clear(" + var + ");\n"
        return code

    def print_test_code(self, outf):
        api = self.api
        params = [
            self.test_param_type(p) + " " + self.test_param_name(p, i)
            for (i, p) in enumerate(api.params)
        ]
        code = "void {}_{}(char *out, {})".format(
            self.test_prefix(), api.name, ", ".join(params)
        )
        code += "{\n"
        code += self.init_vars_from_params()
        code += self.make_api_call()
        code += self.extract_results()
        code += self.clear_local_vars()
        code += "}\n"
        outf.write(code)
        outf.write("\n")


class GMPTest(APITest):
    def __init__(self, gmpapi):
        super(GMPTest, self).__init__(gmpapi)

    def api_call_prefix(self, kind):
        if kind == mpz_t:
            return "mpz_"
        elif kind == mpq_t:
            return "mpq_"
        else:
            raise RuntimeError("Unknown call kind: " + str(kind))

    def mpz_type(self):
        return "mpz_t"

    def mpq_type(self):
        return "mpq_t"


class ImathTest(APITest):
    def __init__(self, gmpapi):
        super(ImathTest, self).__init__(gmpapi)

    def api_call_prefix(self, kind):
        if kind == mpz_t:
            return "impz_"
        elif kind == mpq_t:
            return "impq_"
        else:
            raise RuntimeError("Unknown call kind: " + str(kind))

    def mpz_type(self):
        return "impz_t"

    def mpq_type(self):
        return "impq_t"


def print_gmp_header(outf):
    code = ""
    code += "#include <gmp.h>\n"
    code += "#include <stdio.h>\n"
    code += "#include <string.h>\n"
    code += '#include "gmp_custom_test.c"\n'
    outf.write(code)


def print_imath_header(outf):
    code = ""
    code += "#include <gmp_compat.h>\n"
    code += "#include <stdio.h>\n"
    code += "#include <string.h>\n"
    code += "typedef mpz_t impz_t[1];\n"
    code += "typedef mpq_t impq_t[1];\n"
    code += '#include "imath_custom_test.c"\n'
    outf.write(code)


def print_gmp_tests(outf):
    print_gmp_header(outf)
    for api in gmpapi.apis:
        if not api.custom_test:
            GMPTest(api).print_test_code(outf)


def print_imath_tests(outf):
    print_imath_header(outf)
    for api in gmpapi.apis:
        if not api.custom_test:
            ImathTest(api).print_test_code(outf)


def main():
    test = sys.argv[1]

    if test == "gmp":
        print_gmp_tests(sys.stdout)
    elif test == "imath":
        print_imath_tests(sys.stdout)


if __name__ == "__main__":
    main()
