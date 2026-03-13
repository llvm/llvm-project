# Basic test for substitutions.
#
# RUN: echo %{s:basename} | FileCheck %s --check-prefix=BASENAME
# RUN: echo %{t:stem} %basename_t | FileCheck %s --check-prefix=TMPBASENAME

# BASENAME: substitutions.py
# TMPBASENAME: [[FIRST:[^[:space:]]+]] [[FIRST]]
