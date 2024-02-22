// REQUIRES: x86-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc -cas %t/cas -cas-backend -mccas-casid -triple x86_64-apple-darwin10 %s -filetype=obj -o %t/test.o
// RUN: cd %t && llvm-cas --ingest --cas %t/cas --casid-file %t/test.o > %t/output.casid
// RUN: llvm-cas-object-format --cas %t/cas --materialize-objects --output-prefix %t/output @%t/output.casid 

        .text
_foo:
        ret

_baz:
        call _foo
        ret