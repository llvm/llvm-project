# REQUIRES: x86-registered-target

# RUN: llvm-mc --filetype=obj --triple=i386-w64-windows-gnu %s -o %t.o -g

# RUN: llvm-symbolizer --obj=%t.o 0 1 2 3 4 5 6 7 8 9 10 | FileCheck %s

  .def g
  .scl 2
  .type 32
  .endef
g:
  nop
# CHECK:       {{^g$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:12
# CHECK-EMPTY:

  .def baz
  .scl 2
  .type 32
  .endef
baz:
  nop
# CHECK-NEXT:  {{^baz$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:22
# CHECK-EMPTY:

# extern "C" void c() {} // __cdecl
  .def _c
  .scl 2
  .type 32
  .endef
_c:
  nop
# CHECK-NEXT:  {{^c$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:33
# CHECK-EMPTY:

# extern "C" void __stdcall c1() {}
  .def _c1@0
  .scl 2
  .type 32
  .endef
_c1@0:
  nop
# CHECK-NEXT:  {{^c1$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:44
# CHECK-EMPTY:

# extern "C" void __fastcall c2(void) {}
  .def @c2@0
  .scl 2
  .type 32
  .endef
@c2@0:
  nop
# CHECK-NEXT:  {{^c2$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:55
# CHECK-EMPTY:

# extern "C" void __vectorcall c3(void) {}
  .def c3@@0
  .scl 2
  .type 32
  .endef
c3@@0:
  nop
# CHECK-NEXT:  {{^c3$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:66
# CHECK-EMPTY:

# void f() {} // __cdecl
  .def __Z1fv
  .scl 2
  .type 32
  .endef
__Z1fv:
  nop
# CHECK-NEXT:  {{^f\(\)$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:77
# CHECK-EMPTY:

# void __stdcall f1() {}
  .def __Z2f1v@0
  .scl 2
  .type 32
  .endef
__Z2f1v@0:
  nop
# CHECK-NEXT:  {{^f1\(\)$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:88
# CHECK-EMPTY:

# void __fastcall f2(void) {}
  .def @_Z2f2v@0
  .scl 2
  .type 32
  .endef
@_Z2f2v@0:
  nop
# CHECK-NEXT:  {{^f2\(\)$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:99
# CHECK-EMPTY:

# void __vectorcall f3(void) {}
  .def _Z2f3v@@0
  .scl 2
  .type 32
  .endef
_Z2f3v@@0:
  nop
# CHECK-NEXT:  {{^f3\(\)$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:110
# CHECK-EMPTY:

# Rust
  .def __RNvC1x1y
  .scl 2
  .type 32
  .endef
__RNvC1x1y:
  nop
# CHECK-NEXT:  {{^x::y$}}
# CHECK-NEXT:  symbolize-demangling-mingw32.s:121
# CHECK-EMPTY:
