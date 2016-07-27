; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

;CHECK: [[REGH:r[0-9]]]:[[REGL:[0-9]]] = memd_locked
;CHECK: HIGH([[REGH]])
;CHECK: LOW(r[[REGL]])
define i32 @fred(i64* %free_list_ptr, i32** %item_ptr, i8** %free_item_ptr) nounwind {
entry:
  %free_list_ptr.addr = alloca i64*, align 4
  store i64* %free_list_ptr, i64** %free_list_ptr.addr, align 4
  %0 = load i32*, i32** %item_ptr, align 4
  %1 = call { i64, i32 } asm sideeffect "1:     $0 = memd_locked($5)\0A\09       $1 = HIGH(${0:H}) \0A\09       $1 = add($1,#1) \0A\09       memw($6) = LOW(${0:L}) \0A\09       $0 = combine($7,$1) \0A\09       memd_locked($5,p0) = $0 \0A\09       if !p0 jump 1b\0A\09", "=&r,=&r,=*m,=*m,r,r,r,r,*m,*m,~{p0}"(i64** %free_list_ptr.addr, i8** %free_item_ptr, i64 0, i64* %free_list_ptr, i8** %free_item_ptr, i32* %0, i64** %free_list_ptr.addr, i8** %free_item_ptr) nounwind
  %asmresult1 = extractvalue { i64, i32 } %1, 1
  ret i32 %asmresult1
}
