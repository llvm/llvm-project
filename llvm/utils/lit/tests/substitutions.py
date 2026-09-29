# Basic test for substitutions.
#
# RUN: echo %{s:basename} | FileCheck %s --check-prefix=BASENAME
# RUN: echo %{s:stem} | FileCheck %s --check-prefix=STEM
# RUN: echo %{t:stem} %basename_t | FileCheck %s --check-prefix=TMPBASENAME

# BASENAME: substitutions.py
# STEM: substitutions
# TMPBASENAME: [[FIRST:[^[:space:]]+]] [[FIRST]]
