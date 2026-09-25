; RUN: llc -mtriple=aarch64-linux-gnu -mcpu=generic -pass-remarks-analysis=target-features %s -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mcpu=generic -mattr=+sve2 -pass-remarks-analysis=target-features %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMMAND
; RUN: llc -mtriple=aarch64-linux-gnu -mcpu=generic -pass-remarks-output=- -pass-remarks-filter=target-features %s -o /dev/null | FileCheck %s --check-prefix=YAML

; CPU defaults, implications and explicit disabling are resolved per function.
define void @baseline() { ret void }
define void @enabled() "target-features"="+sve2" { ret void }
define void @disabled() "target-features"="+sve2,-sve" { ret void }
define void @cpu() "target-cpu"="neoverse-v1" { ret void }
define void @baseline_again() { ret void }
define void @"escaped\0Aname"() { ret void }

; Declarations and aliases do not create machine functions.
declare void @declaration()
@alias = alias void (), ptr @baseline

; The baseline is unchanged after functions with custom attributes.
; CHECK-NOT: Enabled features
; CHECK: remark: <unknown>:0:0: Enabled features for @baseline: [[BASELINE:enable-select-opt,ete,fixed-load-latency-4,fp-armv8,fuse-adrp-add,fuse-aes,neon,trbe,use-postra-scheduler]]{{$}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @enabled: {{([^,]+,)*}}fp-armv8,fullfp16,{{([^,]+,)*}}sve,sve2{{(,.*)?$}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @disabled:
; CHECK-NOT: {{(^|[ ,])(sve|sve2)(,|$)}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @cpu: {{([^,]+,)*}}bf16,{{([^,]+,)*}}sve{{(,.*)?$}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @baseline_again: [[BASELINE]]{{$}}
; CHECK-NEXT: remark: <unknown>:0:0: Enabled features for @"escaped\0Aname": [[BASELINE]]{{$}}
; CHECK-NOT: Enabled features

; llc appends -mattr to existing function attributes, so the command line
; re-enables features explicitly disabled by the IR.
; COMMAND: remark: <unknown>:0:0: Enabled features for @baseline: {{([^,]+,)*}}sve,sve2{{(,.*)?$}}
; COMMAND: remark: <unknown>:0:0: Enabled features for @disabled: {{([^,]+,)*}}sve,sve2{{(,.*)?$}}

; Saved analysis remarks expose individual features as structured arguments.
; YAML:      --- !Analysis
; YAML-NEXT: Pass: target-features
; YAML-NEXT: Name: EnabledFeatures
; YAML-NEXT: Function: baseline
; YAML:      - Feature: fp-armv8
; YAML:      Function: enabled
; YAML:      - Feature: fullfp16
; YAML:      - Feature: sve{{$}}
; YAML:      - Feature: sve2{{$}}
; YAML:      Function: disabled
; YAML-NOT:  - Feature: {{sve2?$}}
; YAML:      ...
