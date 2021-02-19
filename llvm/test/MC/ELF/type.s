# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readelf -s - | FileCheck %s

# CHECK:      Symbol table '.symtab' contains 34 entries:
# CHECK-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
# CHECK-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# CHECK-NEXT:   1: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 alias1
# CHECK-NEXT:   2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT     2 alias10
# CHECK-NEXT:   3: 0000000000000000     0 IFUNC   GLOBAL DEFAULT     2 alias11
# CHECK-NEXT:   4: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 alias12
# CHECK-NEXT:   5: 0000000000000000     0 OBJECT  GLOBAL DEFAULT     2 alias2
# CHECK-NEXT:   6: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT     2 alias3
# CHECK-NEXT:   7: 0000000000000000     0 OBJECT  GLOBAL DEFAULT     2 alias4
# CHECK-NEXT:   8: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT     2 alias5
# CHECK-NEXT:   9: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT     2 alias6
# CHECK-NEXT:  10: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 alias7
# CHECK-NEXT:  11: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 alias8
# CHECK-NEXT:  12: 0000000000000000     0 OBJECT  GLOBAL DEFAULT     2 alias9
# CHECK-NEXT:  13: 0000000000000000     0 OBJECT  GLOBAL DEFAULT     2 bar
# CHECK-NEXT:  14: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 foo
# CHECK-NEXT:  15: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 func
# CHECK-NEXT:  16: 0000000000000000     0 IFUNC   GLOBAL DEFAULT     2 ifunc
# CHECK-NEXT:  17: 0000000000000000     0 OBJECT  GLOBAL DEFAULT     2 obj
# CHECK-NEXT:  18: 0000000000000000     0 IFUNC   GLOBAL DEFAULT     2 sym1
# CHECK-NEXT:  19: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 sym10
# CHECK-NEXT:  20: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 sym11
# CHECK-NEXT:  21: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 sym12
# CHECK-NEXT:  22: 0000000000000000     0 IFUNC   GLOBAL DEFAULT     2 sym2
# CHECK-NEXT:  23: 0000000000000000     0 IFUNC   GLOBAL DEFAULT     2 sym3
# CHECK-NEXT:  24: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 sym4
# CHECK-NEXT:  25: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 sym5
# CHECK-NEXT:  26: 0000000000000000     0 OBJECT  GLOBAL DEFAULT     2 sym6
# CHECK-NEXT:  27: 0000000000000000     0 IFUNC   GLOBAL DEFAULT     2 sym7
# CHECK-NEXT:  28: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 sym8
# CHECK-NEXT:  29: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 sym9
# CHECK-NEXT:  30: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 tls
# CHECK-NEXT:  31: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 tls_quoted
# CHECK-NEXT:  32: 0000000000000000     0 TLS     GLOBAL DEFAULT     2 tls_upper_case
# CHECK-NEXT:  33: 0000000000000000     0 OBJECT  UNIQUE DEFAULT     2 zed

// Test that both % and @ are accepted.
        .global foo
        .type foo,%function
foo:

        .global bar
        .type bar,@object
bar:

        .type zed,@gnu_unique_object
zed:

obj:
        .global obj
        .type obj,@object
        .type obj,@notype

func:
        .global func
        .type func,@function
        .type func,@object

ifunc:
        .global ifunc
        .type ifunc,@gnu_indirect_function

tls:
        .global tls
        .type tls,@tls_object
        .type tls,@gnu_indirect_function

// Test that "<type>" is accepted.
tls_quoted:
        .global tls_quoted
        .type tls_quoted,"tls_object"

// Test that "<type>" is accepted.
tls_upper_case:
        .global tls_upper_case
        .type tls_upper_case,STT_TLS

// Test that .set doesnt downgrade the type:
// IFUNC > FUNC > OBJECT > NOTYPE
// TLS_OBJECT > OBJECT > NOTYPE
// also TLS_OBJECT is incompatible with IFUNC and FUNC

        .global sym1
        .type sym1, @gnu_indirect_function
alias1:
        .global alias1
        .type alias1, @function
        .set sym1, alias1

        .global sym2
        .type sym2, @gnu_indirect_function
alias2:
        .global alias2
        .type alias2, @object
        .set sym2, alias2

        .global sym3
        .type sym3, @gnu_indirect_function
alias3:
        .global alias3
        .type alias3, @notype
        .set sym3, alias3

        .global sym4
        .type sym4, @function
alias4:
        .global alias4
        .type alias4, @object
        .set sym4, alias4

        .global sym5
        .type sym5, @function
alias5:
        .global alias5
        .type alias5, @notype
        .set sym5, alias5

        .global sym6
        .type sym6, @object
alias6:
        .global alias6
        .type alias6, @notype
        .set sym6, alias6

        .global sym7
        .type sym7, @gnu_indirect_function
alias7:
        .global alias7
        .type alias7, @tls_object
        .set sym7, alias7

        .global sym8
        .type sym8, @function
        .global alias8
alias8:
        .type alias8, @tls_object
        .set sym8, alias8

        .global sym9
        .type sym9, @tls_object
alias9:
        .global alias9
        .type alias9, @object
        .set sym9, alias9

        .global sym10
        .type sym10, @tls_object
alias10:
        .global alias10
        .type alias10, @notype
        .set sym10, alias10

        .global sym11
        .type sym11, @tls_object
alias11:
        .global alias11
        .type alias11, @gnu_indirect_function
        .set sym11, alias11

        .global sym12
        .type sym12, @tls_object
alias12:        
        .global alias12
        .type alias12, @function
        .set sym12, alias12
