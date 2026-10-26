; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; PTX can represent cyclic global references as relocations when ptxas has
; already seen a compatible declaration for the referenced symbol.
; Consumer appears first in IR to also check dependency-first SCC ordering.
; The cycles below are mutually disconnected, so emitting all of them also
; checks that the synthetic root reaches distinct cyclic SCCs.

; PTX32:      .extern .global .align 4 .u32 a;
; PTX32-NEXT: .extern .global .align 4 .u32 b;
; PTX32-NEXT: .visible .global .align 4 .u32 a = b;
; PTX32-NEXT: .visible .global .align 4 .u32 b = a;
; PTX32-NEXT: .visible .global .align 4 .u32 consumer = a;
; PTX64:      .extern .global .align 8 .u64 a;
; PTX64-NEXT: .extern .global .align 8 .u64 b;
; PTX64-NEXT: .visible .global .align 8 .u64 a = b;
; PTX64-NEXT: .visible .global .align 8 .u64 b = a;
; PTX64-NEXT: .visible .global .align 8 .u64 consumer = a;
@consumer = addrspace(1) global ptr addrspace(1) @a
@a = addrspace(1) global ptr addrspace(1) @b
@b = addrspace(1) global ptr addrspace(1) @a

; If a cycle includes an internal global, only the externally-visible global is
; forward-declared. The definitions are still ordered so the internal symbol is
; defined before it is referenced.

; PTX32:      .extern .global .align 4 .u32 c;
; PTX32-NEXT: .global .align 4 .u32 d = c;
; PTX32-NEXT: .visible .global .align 4 .u32 c = d;
; PTX64:      .extern .global .align 8 .u64 c;
; PTX64-NEXT: .global .align 8 .u64 d = c;
; PTX64-NEXT: .visible .global .align 8 .u64 c = d;
@c = addrspace(1) global ptr addrspace(1) @d
@d = internal addrspace(1) global ptr addrspace(1) @c

; One forward declaration can break a larger cycle containing multiple
; internal globals. This requires multiple ready-list updates to order the
; definitions as three_c, three_b, three_a.

; PTX32:      .extern .global .align 4 .u32 three_a;
; PTX32-NEXT: .global .align 4 .u32 three_c = three_a;
; PTX32-NEXT: .global .align 4 .u32 three_b = three_c;
; PTX32-NEXT: .visible .global .align 4 .u32 three_a = three_b;
; PTX64:      .extern .global .align 8 .u64 three_a;
; PTX64-NEXT: .global .align 8 .u64 three_c = three_a;
; PTX64-NEXT: .global .align 8 .u64 three_b = three_c;
; PTX64-NEXT: .visible .global .align 8 .u64 three_a = three_b;
@three_a = addrspace(1) global ptr addrspace(1) @three_b
@three_b = internal addrspace(1) global ptr addrspace(1) @three_c
@three_c = internal addrspace(1) global ptr addrspace(1) @three_a

; Non-local weak definitions can also resolve compatible .extern declarations.

; PTX32:      .extern .global .align 4 .u32 weak_a;
; PTX32-NEXT: .extern .global .align 4 .u32 weak_b;
; PTX32-NEXT: .weak .global .align 4 .u32 weak_a = weak_b;
; PTX32-NEXT: .weak .global .align 4 .u32 weak_b = weak_a;
; PTX64:      .extern .global .align 8 .u64 weak_a;
; PTX64-NEXT: .extern .global .align 8 .u64 weak_b;
; PTX64-NEXT: .weak .global .align 8 .u64 weak_a = weak_b;
; PTX64-NEXT: .weak .global .align 8 .u64 weak_b = weak_a;
@weak_a = weak addrspace(1) global ptr addrspace(1) @weak_b
@weak_b = weak addrspace(1) global ptr addrspace(1) @weak_a

; Forward declarations retain the globals' PTX state space.

; PTX32:      .extern .const .align 4 .u32 const_a;
; PTX32-NEXT: .extern .const .align 4 .u32 const_b;
; PTX32-NEXT: .visible .const .align 4 .u32 const_a = const_b;
; PTX32-NEXT: .visible .const .align 4 .u32 const_b = const_a;
; PTX64:      .extern .const .align 8 .u64 const_a;
; PTX64-NEXT: .extern .const .align 8 .u64 const_b;
; PTX64-NEXT: .visible .const .align 8 .u64 const_a = const_b;
; PTX64-NEXT: .visible .const .align 8 .u64 const_b = const_a;
@const_a = addrspace(4) constant ptr addrspace(4) @const_b
@const_b = addrspace(4) constant ptr addrspace(4) @const_a
