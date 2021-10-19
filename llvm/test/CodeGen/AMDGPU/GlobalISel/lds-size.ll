; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 < %S/../lds-size.ll | FileCheck -check-prefix=ALL -check-prefix=HSA %S/../lds-size.ll
