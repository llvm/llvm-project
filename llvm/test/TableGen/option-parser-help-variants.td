// RUN: not llvm-tblgen -gen-opt-parser-defs -I %p/../../include \
// RUN:   -DTOO_MANY_VARIANTS %s 2>&1 | FileCheck %s --check-prefix=VARIANTS
// RUN: not llvm-tblgen -gen-opt-parser-defs -I %p/../../include \
// RUN:   -DTOO_MANY_VISIBILITIES %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=VISIBILITIES

include "llvm/Option/OptParser.td"

def V1 : OptionVisibility;
def V2 : OptionVisibility;
def V3 : OptionVisibility;

#ifdef TOO_MANY_VARIANTS
def bad_variants : Flag<["--"], "bad-variants"> {
  let HelpTextsForVariants = [
    HelpTextVariant<[V1], "one">,
    HelpTextVariant<[V2], "two">
  ];
}

// VARIANTS: error: at most one HelpTextForVariants entry is supported
#endif

#ifdef TOO_MANY_VISIBILITIES
def bad_visibilities : Flag<["--"], "bad-visibilities">,
  HelpTextForVariants<[V1, V2, V3], "bad help">;

// VISIBILITIES: error: at most two visibilities are supported per HelpTextForVariants entry
#endif
