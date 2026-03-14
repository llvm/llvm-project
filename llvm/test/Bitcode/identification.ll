; Check that a block "IDENTIFICATION_BLOCK_ID" is emitted.
;RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s
;CHECK: <IDENTIFICATION_BLOCK_ID
;CHECK-NEXT: <STRING
;CHECK-NEXT: <EPOCH
;CHECK-NEXT: </IDENTIFICATION_BLOCK_ID
