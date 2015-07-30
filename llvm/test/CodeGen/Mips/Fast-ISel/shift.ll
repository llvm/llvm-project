; RUN: llc -march=mipsel -mcpu=mips32r2 -O0 -fast-isel=true -filetype=obj %s -o - \
; RUN:   | llvm-objdump -arch mipsel -mcpu=mips32r2 -d - | FileCheck %s

; This test checks that encoding for srl is correct when fast-isel for mips32r2 is used.

%struct.s = type { [4 x i8], i32 }

define i32 @main() nounwind uwtable {
entry:
  %foo = alloca %struct.s, align 4
  %0 = bitcast %struct.s* %foo to i32*
  %bf.load = load i32, i32* %0, align 4
  %bf.lshr = lshr i32 %bf.load, 2
  %cmp = icmp ne i32 %bf.lshr, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:
  unreachable

if.end:
  ret i32 0
}

; CHECK: srl    ${{[0-9]+}}, ${{[0-9]+}}, {{[0-9]+}}
