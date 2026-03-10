; REQUIRES: x86_64-linux

; Test that each remark category can be enabled independently.

; -Rpass-missed: only unmatched functions
; RUN: opt < %S/pseudo-probe-stale-profile-renaming.ll -passes=sample-profile \
; RUN:   -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-renaming.prof \
; RUN:   --salvage-stale-profile --salvage-unused-profile \
; RUN:   -pass-remarks-missed=sample-profile-matcher \
; RUN:   --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 \
; RUN:   --func-profile-similarity-threshold=70 -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSED --allow-empty

; MISSED-NOT: sample profile matched
; MISSED-NOT: sample profile recovered

; -Rpass: matched and recovered functions
; RUN: opt < %S/pseudo-probe-stale-profile-renaming.ll -passes=sample-profile \
; RUN:   -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-renaming.prof \
; RUN:   --salvage-stale-profile --salvage-unused-profile \
; RUN:   -pass-remarks=sample-profile-matcher \
; RUN:   --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 \
; RUN:   --func-profile-similarity-threshold=70 -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASS

; PASS-DAG: remark: {{.*}}: sample profile recovered for new_foo matched to profile 'foo'
; PASS-DAG: remark: {{.*}}: sample profile recovered for new_block_only matched to profile 'block_only'
; PASS-DAG: remark: {{.*}}: sample profile matched for main with {{[0-9]+}} total samples
; PASS-DAG: remark: {{.*}}: sample profile matched for bar with {{[0-9]+}} total samples
; PASS-NOT: no sample profile matched

; All categories together
; RUN: opt < %S/pseudo-probe-stale-profile-renaming.ll -passes=sample-profile \
; RUN:   -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-renaming.prof \
; RUN:   --salvage-stale-profile --salvage-unused-profile \
; RUN:   -pass-remarks=sample-profile-matcher -pass-remarks-missed=sample-profile-matcher \
; RUN:   -pass-remarks-analysis=sample-profile-matcher \
; RUN:   --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 \
; RUN:   --func-profile-similarity-threshold=70 -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=ALL

; ALL-DAG: remark: {{.*}}: sample profile recovered for new_foo matched to profile 'foo'
; ALL-DAG: remark: {{.*}}: sample profile recovered for new_block_only matched to profile 'block_only'
; ALL-DAG: remark: {{.*}}: sample profile matched for main with {{[0-9]+}} total samples
; ALL-DAG: remark: {{.*}}: sample profile matched for bar with {{[0-9]+}} total samples
; ALL-NOT: no sample profile matched for new_foo
; ALL-NOT: no sample profile matched for new_block_only
