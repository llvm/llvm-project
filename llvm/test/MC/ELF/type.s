// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols - | FileCheck %s

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

// CHECK:        Symbol {
// CHECK:          Name: bar
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: func
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: ifunc
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: GNU_IFunc
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: obj
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym1
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: GNU_IFunc (0xA)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym10
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: TLS (0x6)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym11
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: TLS (0x6)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym12
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: TLS (0x6)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym2
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: GNU_IFunc (0xA)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym3
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: GNU_IFunc (0xA)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym4
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: Function (0x2)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym5
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: Function (0x2)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym6
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: Object (0x1)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym7
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: GNU_IFunc (0xA)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym8
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: Function (0x2)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym9
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global (0x1)
// CHECK-NEXT:    Type: TLS (0x6)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: tls
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: tls_quoted
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: tls_upper_case
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: zed
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Unique (0xA)
// CHECK-NEXT:     Type: Object (0x1)
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT: ]
