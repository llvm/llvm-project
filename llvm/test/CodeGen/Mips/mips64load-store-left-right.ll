; RUN: llc -march=mips64el -mcpu=mips64 -mattr=n64 < %s | FileCheck  -check-prefix=EL %s
; RUN: llc -march=mips64 -mcpu=mips64 -mattr=n64 < %s | FileCheck  -check-prefix=EB %s

%struct.SLL = type { i64 }
%struct.SI = type { i32 }
%struct.SUI = type { i32 }

@sll = common global %struct.SLL zeroinitializer, align 1
@si = common global %struct.SI zeroinitializer, align 1
@sui = common global %struct.SUI zeroinitializer, align 1

define i64 @foo_load_ll() nounwind readonly {
entry:
; EL: ldl $[[R0:[0-9]+]], 7($[[R1:[0-9]+]])
; EL: ldr $[[R0]], 0($[[R1]])
; EB: ldl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; EB: ldr $[[R0]], 7($[[R1]])

  %0 = load i64* getelementptr inbounds (%struct.SLL* @sll, i64 0, i32 0), align 1
  ret i64 %0
}

define i64 @foo_load_i() nounwind readonly {
entry:
; EL: lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; EL: lwr $[[R0]], 0($[[R1]])
; EB: lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; EB: lwr $[[R0]], 3($[[R1]])

  %0 = load i32* getelementptr inbounds (%struct.SI* @si, i64 0, i32 0), align 1
  %conv = sext i32 %0 to i64
  ret i64 %conv
}

define i64 @foo_load_ui() nounwind readonly {
entry:
; EL: lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; EL: lwr $[[R0]], 0($[[R1]])
; EL: daddiu $[[R2:[0-9]+]], $zero, 1
; EL: dsll   $[[R3:[0-9]+]], $[[R2]], 32
; EL: daddiu $[[R4:[0-9]+]], $[[R3]], -1
; EL: and    ${{[0-9]+}}, $[[R0]], $[[R4]]
; EB: lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; EB: lwr $[[R0]], 3($[[R1]])


  %0 = load i32* getelementptr inbounds (%struct.SUI* @sui, i64 0, i32 0), align 1
  %conv = zext i32 %0 to i64
  ret i64 %conv
}

define void @foo_store_ll(i64 %a) nounwind {
entry:
; EL: sdl $[[R0:[0-9]+]], 7($[[R1:[0-9]+]])
; EL: sdr $[[R0]], 0($[[R1]])
; EB: sdl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; EB: sdr $[[R0]], 7($[[R1]])

  store i64 %a, i64* getelementptr inbounds (%struct.SLL* @sll, i64 0, i32 0), align 1
  ret void
}

define void @foo_store_i(i32 %a) nounwind {
entry:
; EL: swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; EL: swr $[[R0]], 0($[[R1]])
; EB: swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; EB: swr $[[R0]], 3($[[R1]])

  store i32 %a, i32* getelementptr inbounds (%struct.SI* @si, i64 0, i32 0), align 1
  ret void
}

