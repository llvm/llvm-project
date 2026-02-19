; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefixes=CHECK,DAG
; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -fast-isel -fast-isel-abort=1 | FileCheck %s --check-prefixes=CHECK,FAST
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefixes=CHECK,DAG
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -fast-isel -fast-isel-abort=1 | FileCheck %s --check-prefixes=CHECK,FAST

; Test that extending loads are assembled properly.

; CHECK-LABEL: sext_i8_i32:
; DAG: i32.load8_s $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load8_u   $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i32.extend8_s $push[[EXT:[0-9]+]]=, $pop[[L]]
; FAST-NEXT: return        $pop[[EXT]]
define i32 @sext_i8_i32(ptr %p) {
  %v = load i8, ptr %p
  %e = sext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i8_i32:
; DAG: i32.load8_u $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:       i32.load8_u $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT:       i32.const   $push[[C:[0-9]+]]=, 255
; FAST-NEXT:       i32.and     $push[[R:[0-9]+]]=, $pop[[L]], $pop[[C]]
; FAST-NEXT:       return      $pop[[R]]
define i32 @zext_i8_i32(ptr %p) {
  %v = load i8, ptr %p
  %e = zext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i16_i32:
; DAG: i32.load16_s $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load16_u   $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i32.extend16_s $push[[EXT:[0-9]+]]=, $pop[[L]]
; FAST-NEXT: return         $pop[[EXT]]
define i32 @sext_i16_i32(ptr %p) {
  %v = load i16, ptr %p
  %e = sext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i16_i32:
; DAG: i32.load16_u $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load16_u $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i32.const    $push[[C:[0-9]+]]=, 65535
; FAST-NEXT: i32.and      $push[[R:[0-9]+]]=, $pop[[L]], $pop[[C]]
; FAST-NEXT: return       $pop[[R]]
define i32 @zext_i16_i32(ptr %p) {
  %v = load i16, ptr %p
  %e = zext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i8_i64:
; DAG: i64.load8_s $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load8_u      $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i64.extend_i32_u $push[[EXT32:[0-9]+]]=, $pop[[L]]
; FAST-NEXT: i64.extend8_s    $push[[EXT8:[0-9]+]]=, $pop[[EXT32]]
; FAST-NEXT: return           $pop[[EXT8]]
define i64 @sext_i8_i64(ptr %p) {
  %v = load i8, ptr %p
  %e = sext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i8_i64:
; DAG: i64.load8_u $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load8_u      $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i32.const        $push[[C:[0-9]+]]=, 255
; FAST-NEXT: i32.and          $push[[AND:[0-9]+]]=, $pop[[L]], $pop[[C]]
; FAST-NEXT: i64.extend_i32_u $push[[EXT:[0-9]+]]=, $pop[[AND]]
; FAST-NEXT: return           $pop[[EXT]]
define i64 @zext_i8_i64(ptr %p) {
  %v = load i8, ptr %p
  %e = zext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i16_i64:
; DAG: i64.load16_s $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load16_u     $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i64.extend_i32_u $push[[EXT32:[0-9]+]]=, $pop[[L]]
; FAST-NEXT: i64.extend16_s   $push[[EXT16:[0-9]+]]=, $pop[[EXT32]]
; FAST-NEXT: return           $pop[[EXT16]]
define i64 @sext_i16_i64(ptr %p) {
  %v = load i16, ptr %p
  %e = sext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i16_i64:
; DAG: i64.load16_u $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load16_u     $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i32.const        $push[[C:[0-9]+]]=, 65535
; FAST-NEXT: i32.and          $push[[AND:[0-9]+]]=, $pop[[L]], $pop[[C]]
; FAST-NEXT: i64.extend_i32_u $push[[EXT:[0-9]+]]=, $pop[[AND]]
; FAST-NEXT: return           $pop[[EXT]]
define i64 @zext_i16_i64(ptr %p) {
  %v = load i16, ptr %p
  %e = zext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i32_i64:
; DAG: i64.load32_s $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load         $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i64.extend_i32_s $push[[EXT:[0-9]+]]=, $pop[[L]]
; FAST-NEXT: return           $pop[[EXT]]
define i64 @sext_i32_i64(ptr %p) {
  %v = load i32, ptr %p
  %e = sext i32 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i32_i64:
; DAG: i64.load32_u $push0=, 0($0){{$}}
; DAG-NEXT: return $pop0{{$}}
; FAST:      i32.load         $push[[L:[0-9]+]]=, 0($0)
; FAST-NEXT: i64.extend_i32_u $push[[EXT:[0-9]+]]=, $pop[[L]]
; FAST-NEXT: return           $pop[[EXT]]
define i64 @zext_i32_i64(ptr %p) {
  %v = load i32, ptr %p
  %e = zext i32 %v to i64
  ret i64 %e
}
