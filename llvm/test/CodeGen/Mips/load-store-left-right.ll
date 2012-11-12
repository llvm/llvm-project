; RUN: llc -march=mipsel < %s | FileCheck  -check-prefix=EL %s
; RUN: llc -march=mips < %s | FileCheck  -check-prefix=EB %s

%struct.SI = type { i32 }

@si = common global %struct.SI zeroinitializer, align 1

define i32 @foo_load_i() nounwind readonly {
entry:
; EL: lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; EL: lwr $[[R0]], 0($[[R1]])
; EB: lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; EB: lwr $[[R0]], 3($[[R1]])

  %0 = load i32* getelementptr inbounds (%struct.SI* @si, i32 0, i32 0), align 1
  ret i32 %0
}

define void @foo_store_i(i32 %a) nounwind {
entry:
; EL: swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; EL: swr $[[R0]], 0($[[R1]])
; EB: swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; EB: swr $[[R0]], 3($[[R1]])

  store i32 %a, i32* getelementptr inbounds (%struct.SI* @si, i32 0, i32 0), align 1
  ret void
}

