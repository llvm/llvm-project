#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cat \
    ${PROJECT_DIR}/flang/tools/shared/rtlRtnsDesc.h \
    ${PROJECT_DIR}/flang/tools/shared/rtlRtns.h \
    ${PROJECT_DIR}/flang/tools/shared/rtlRtns.c \
    - << END |

#include <vector>
#include <bitset>
#include <string>
#include <algorithm>

int main(void) {
    std::vector<std::string> non_variadic_routines;
    std::bitset<END_OF_FTNIO> has_variadic_impl;

    for (int rteRtn : {
        RTE_comm_free, RTE_conformable_dnv, RTE_conformable_ndv, RTE_conformable_nnv, RTE_extends_type_of, RTE_freen, RTE_instance,
        RTE_ksize, RTE_lb, RTE_lb1, RTE_lb2, RTE_lb4, RTE_lb8, RTE_lba, RTE_lba1, RTE_lba2, RTE_lba4, RTE_lba8, RTE_lbaz, RTE_lbaz1,
        RTE_lbaz2, RTE_lbaz4, RTE_lbaz8, RTE_lbound, RTE_lbound1, RTE_lbound2, RTE_lbound4, RTE_lbound8, RTE_lbounda, RTE_lbounda1,
        RTE_lbounda2, RTE_lbounda4, RTE_lbounda8, RTE_lboundaz, RTE_lboundaz1, RTE_lboundaz2, RTE_lboundaz4, RTE_lboundaz8, RTE_max,
        RTE_max, RTE_maxval_scatterx, RTE_min, RTE_min, RTE_minval_scatterx, RTE_nstr_copy, RTE_nstr_copy_klen, RTE_olap_cshift,
        RTE_olap_shift, RTE_permute_section, RTE_poly_element_addr, RTE_processors, RTE_ptr_fix_assumeshp, RTE_ptr_shape_assn,
        RTE_ptr_shape_assnx, RTE_qopy_in, RTE_realign, RTE_redistribute, RTE_same_intrin_type_as, RTE_same_type_as, RTE_sect, RTE_shape,
        RTE_shape1, RTE_shape2, RTE_shape4, RTE_shape8, RTE_size, RTE_str_copy, RTE_str_copy_klen, RTE_template, RTE_ub, RTE_ub1,
        RTE_ub2, RTE_ub4, RTE_ub8, RTE_uba, RTE_uba1, RTE_uba2, RTE_uba4, RTE_uba8, RTE_ubaz, RTE_ubaz1, RTE_ubaz2, RTE_ubaz4, RTE_ubaz8,
        RTE_templateDsc, RTE_ubound, RTE_ubound1, RTE_ubound2, RTE_ubound4, RTE_ubound8, RTE_ubounda, RTE_ubounda1, RTE_ubounda2, RTE_ubounda4,
        RTE_ubounda8, RTE_uboundaz, RTE_uboundaz1, RTE_uboundaz2, RTE_uboundaz4, RTE_uboundaz8,
    }) {
        has_variadic_impl.set(rteRtn);
    }

    for (int rteRtn = RTE_no_rtn + 1; rteRtn < END_OF_FTNIO; rteRtn++) {
        if (rteRtn == END_OF_PFX_F90 ||
            rteRtn == END_OF_PFX_FTN ||
            rteRtn == END_OF_IO)
            continue;

        if (has_variadic_impl.test(rteRtn))
            continue;

        std::string r(mkRteRtnNm((FtnRtlEnum)rteRtn));
        non_variadic_routines.push_back(r);
        if (ftnRtlRtns[rteRtn].I8Descr)
            non_variadic_routines.push_back(r + "_i8");
        if (strstr(ftnRtlRtns[rteRtn].largeRetValPrefix, "k"))
            non_variadic_routines.push_back(r + ftnRtlRtns[rteRtn].largeRetValPrefix);
    }

    for (const std::string &intrinsic : {
        "ftn_i_jishft", "ftn_i_shift",
        "ftn_i_rmin", "ftn_i_rmax", "ftn_i_dmax",
        "ftn_i_dmin", "ftn_i_isign", "ftn_i_sign",
        "ftn_i_dsign",
        "ftn_i_dim", "ftn_i_idim", "ftn_i_ddim",
    }) {
        non_variadic_routines.push_back(intrinsic);
    }

    for (const std::string &math_intrinsic : {
        "__mth_i_ileadz", "__mth_i_ileadzi", "__mth_i_kleadz",
        "__mth_i_ipopcnt", "__mth_i_ipopcnti", "__mth_i_ipoppar", "__mth_i_ipoppari", "__mth_i_kpopcnt", "__mth_i_kpoppar",
        "__mth_i_bessel_j0", "__mth_i_bessel_j1", "__mth_i_bessel_jn",
        "__mth_i_bessel_y0", "__mth_i_bessel_y1", "__mth_i_bessel_yn",
        "__mth_i_dbessel_j0", "__mth_i_dbessel_j1", "__mth_i_dbessel_jn",
        "__mth_i_dbessel_y0", "__mth_i_dbessel_y1", "__mth_i_dbessel_yn",
        "__mth_i_idnint", "__mth_i_kidnnt", "__mth_i_knint",
        "__mth_i_kmul", "__mth_i_kdiv", "__mth_i_ukdiv",
        "__mth_i_kcmp", "__mth_i_kucmp", "__mth_i_kcmpz", "__mth_i_kucmpz",
        "__mth_i_krshift", "__mth_i_klshift", "__mth_i_kurshift", "__mth_i_kicshft", "__mth_i_ukicshft", "__mth_i_kishft",
        "__mth_i_nint",
        "__mth_i_dexp",
    }) {
        non_variadic_routines.push_back(math_intrinsic);
    }

    std::sort(non_variadic_routines.begin(), non_variadic_routines.end());

    printf("/* AUTO-GENERATED FILE, DO NOT EDIT */\n");
    printf("/* List of white-listed Fortran built-ins which have a non-variadic definition */\n");
    for (const auto &r : non_variadic_routines) {
        printf("\"%s\",\n", r.c_str());
    }

    return 0;
}

END
    grep -v "\s*#\s*include.*\"" | \
g++ \
    -Wno-write-strings \
    -include stdio.h -include stdbool.h \
    -Dassert= -DERR_Severe -DTRUE=true -DFALSE=false \
    -DISZ_T=size_t -DLOGICAL=int -D'UINT=unsigned int' \
    -D'XBIT(n, m)=false' -D'MAXIDLEN=163' \
    -xc++ - \
    -o main && \
    ./main
