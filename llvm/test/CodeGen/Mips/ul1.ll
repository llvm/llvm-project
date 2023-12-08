; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16
%struct.ua = type <{ i16, i32 }>

@foo = common global %struct.ua zeroinitializer, align 1

define i32 @main() nounwind {
entry:
  store i32 10, ptr getelementptr inbounds (%struct.ua, ptr @foo, i32 0, i32 1), align 1
; 16:   sb  ${{[0-9]+}}, {{[0-9]+}}(${{[0-9]+}})
; 16:   sb  ${{[0-9]+}}, {{[0-9]+}}(${{[0-9]+}})
; 16:   sb  ${{[0-9]+}}, {{[0-9]+}}(${{[0-9]+}})
; 16:   sb  ${{[0-9]+}}, {{[0-9]+}}(${{[0-9]+}})
  ret i32 0
}

