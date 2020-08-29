#include "P2BaseInfo.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
// using namespace P2;

// convert a encoding to a condition string
const char *P2::cond_string_lut[] = {
    "\t_ret_\t",
    "if_nc_and_nz\t",
    "if_nc_and_z\t",
    "\tif_nc\t",
    "if_c_and_nz\t",
    "\tif_nz\t",
    "if_c_ne_z\t",
    "if_nc_or_nz\t",
    "if_c_and_z\t",
    "if_c_eq_z\t",
    "\tif_z\t",
    "if_nc_or_z\t",
    "\tif_c\t",
    "if_c_or_nz\t",
    "if_c_or_z\t",
    "\t\t"
};

const char *P2::effect_string_lut[] = {
    "",
    "wz",
    "wc",
    "wcz"
};

// TODO add the other flags that have the same values
// map a string to a condition encoding/immediate
std::map<StringRef, int> P2::cond_string_map = {
    {"_ret_",           0x0},
    {"if_nc_and_nz",    0x1},
    {"if_nc_and_z",     0x2},
    {"if_nc",           0x3},
    {"if_c_and_nz",     0x4},
    {"if_nz",           0x5},
    {"if_c_ne_z",       0x6},
    {"if_nc_or_nz",     0x7},
    {"if_c_and_z",      0x8},
    {"if_c_eq_z",       0x9},
    {"if_z",            0xa},
    {"if_nc_or_z",      0xb},
    {"if_c",            0xc},
    {"if_c_or_nz",      0xd},
    {"if_c_or_z",       0xe},
    {"",                0xf}
};

// map a string to a condition encoding/immediate
std::map<StringRef, int> P2::effect_string_map = {
    {"",        0x0},
    {"wz",      0x1},
    {"wc",      0x2},
    {"wcz",     0x3}
};