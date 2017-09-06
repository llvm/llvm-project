# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
# RUN: obj2yaml %t.o | FileCheck %s

# CHECK: Sections:
# CHECK: - Name:            .group{{$}}
# CHECK:   Members:
# CHECK:     - SectionOrType:   .text.foo{{$}}
# CHECK:     - SectionOrType:   .rela.text.foo{{$}}
# CHECK: - Name:            .text.foo{{$}}
# CHECK: - Name:            .rela.text.foo{{$}}
# CHECK:   Info:            .text.foo{{$}}
# CHECK: - Name:            .group1{{$}}
# CHECK:   Members:
# CHECK:     - SectionOrType:   .text.foo2{{$}}
# CHECK:     - SectionOrType:   .rela.text.foo3{{$}}
# CHECK: - Name:            .text.foo2{{$}}
# CHECK: - Name:            .rela.text.foo3{{$}}
# CHECK:   Info:            .text.foo2{{$}}
# CHECK: Symbols:
# CHECK:   Section:         .group{{$}}
# CHECK:   Section:         .group1{{$}}


        .section        .text.foo,"axG",@progbits,sym1,comdat
        .quad undef

        .section        .text.foo,"axG",@progbits,sym2,comdat
        .quad undef
