; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+2]]:41: error: missing required field 'value'
!0 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter,
                               type: !{})
