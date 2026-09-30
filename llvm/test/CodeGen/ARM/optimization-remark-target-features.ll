; RUN: llc -mtriple=armv7-linux-gnueabi -mcpu=generic -pass-remarks-analysis=target-features %s -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=armv7-linux-gnueabi -mcpu=generic -mattr=+neon -pass-remarks-analysis=target-features %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMMAND

; CPU defaults, implications and explicit disabling are resolved per function.
define void @baseline() { ret void }
define void @enabled() "target-features"="+neon" { ret void }
define void @disabled() "target-features"="+neon,-vfp3" { ret void }
define void @cpu() "target-cpu"="cortex-a8" { ret void }
define void @baseline_again() { ret void }
define void @"escaped\0Aname"() { ret void }

; Declarations and aliases do not create machine functions.
declare void @declaration()
@alias = alias void (), ptr @baseline

; The baseline is unchanged after functions with custom attributes.
; CHECK-NOT: Enabled features
; CHECK: remark: <unknown>:0:0: Enabled features for @baseline: [[BASELINE:.*]]
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @enabled: {{([^,]+,)*}}neon,{{([^,]+,)*}}vfp2,{{([^,]+,)*}}vfp3{{(,.*)?$}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @disabled:
; CHECK-NOT: {{(^|[ ,])(neon|vfp3)(,|$)}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @cpu: {{([^,]+,)*}}neon,{{([^,]+,)*}}vfp3{{(,.*)?$}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @baseline_again: [[BASELINE]]{{$}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @"escaped\0Aname": [[BASELINE]]{{$}}
; CHECK-NOT: Enabled features

; llc appends -mattr to existing function attributes, so the command line
; re-enables features explicitly disabled by the IR.
; COMMAND: remark: <unknown>:0:0: Enabled features for @baseline: {{([^,]+,)*}}neon{{(,.*)?$}}
; COMMAND: remark: <unknown>:0:0: Enabled features for @disabled: {{([^,]+,)*}}neon{{(,.*)?$}}
