; RUN: opt < %s -passes='require<profile-summary>,function(lower-allow-check<cutoffs[7]=0;cutoffs[8]=1>)' -S -o - -print-pipeline-passes | FileCheck %s

; CHECK: lower-allow-check<cutoffs[8]=1>
