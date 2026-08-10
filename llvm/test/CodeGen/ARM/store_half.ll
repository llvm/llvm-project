; RUN: llc < %s -mtriple=thumbebv8.2a-arm-none-eabi -mattr=+fullfp16 -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=thumbv8.2a-arm-none-eabi -mattr=+fullfp16 -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=armebv8.2a-arm-none-eabi -mattr=+fullfp16 -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=armv8.2a-arm-none-eabi -mattr=+fullfp16 -filetype=obj -o /dev/null

define void @woah(ptr %waythere) {
  store half 0xHE110, ptr %waythere
  ret void
}
