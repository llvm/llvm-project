# RUN: %{lit} -a --filter-requires=Half --filter-out=regex %S/Inputs/filter-requires | FileCheck %s --check-prefix=HALF
# HALF: UNSUPPORTED{{:}} filter-requires :: and.txt
# HALF: UNSUPPORTED{{:}} filter-requires :: base.txt
# HALF: PASS: filter-requires :: features.txt
# HALF: PASS: filter-requires :: half.txt
# HALF: UNSUPPORTED{{:}} filter-requires :: negative.txt
# HALF: UNSUPPORTED{{:}} filter-requires :: nested.txt
# HALF: PASS: filter-requires :: or.txt
# HALF: UNSUPPORTED{{:}} filter-requires :: true.txt
# HALF: UNSUPPORTED{{:}} filter-requires :: unsupported.txt
# HALF: XFAIL{{:}} filter-requires :: xfail.txt

# RUN: %{lit} -a --filter-requires="Int16, Half" --filter-out=regex %S/Inputs/filter-requires | FileCheck %s --check-prefix=AND
# AND: PASS: filter-requires :: and.txt
# AND: UNSUPPORTED{{:}} filter-requires :: half.txt
# AND: PASS: filter-requires :: nested.txt
# AND: UNSUPPORTED{{:}} filter-requires :: or.txt

# RUN: %{lit} -a --filter-requires="Half || Int16" --filter-out=regex %S/Inputs/filter-requires | FileCheck %s --check-prefix=OR
# OR: UNSUPPORTED{{:}} filter-requires :: and.txt
# OR: PASS: filter-requires :: half.txt
# OR: UNSUPPORTED{{:}} filter-requires :: nested.txt
# OR: PASS: filter-requires :: or.txt

# RUN: %{lit} -a --filter-requires="Half && !Double" --filter-out=regex %S/Inputs/filter-requires | FileCheck %s --check-prefix=NEGATIVE
# NEGATIVE: PASS: filter-requires :: half.txt
# NEGATIVE: PASS: filter-requires :: negative.txt
# NEGATIVE: UNSUPPORTED{{:}} filter-requires :: nested.txt

# RUN: %{lit} -a --filter-requires="Half && Int16 && !Double" --filter-out=regex %S/Inputs/filter-requires | FileCheck %s --check-prefix=NESTED
# NESTED: PASS: filter-requires :: and.txt
# NESTED: UNSUPPORTED{{:}} filter-requires :: half.txt
# NESTED: PASS: filter-requires :: nested.txt
# NESTED: UNSUPPORTED{{:}} filter-requires :: or.txt

# RUN: %{lit} -a --filter-requires=Base %S/Inputs/filter-requires | FileCheck %s --check-prefix=BASE
# BASE: PASS: filter-requires :: base.txt
# BASE: UNSUPPORTED{{:}} filter-requires :: half.txt
# BASE: UNSUPPORTED{{:}} filter-requires :: true.txt
# BASE: Passed{{ *}}: 1

# RUN: %{lit} -a --filter-requires=true --filter-out=regex %S/Inputs/filter-requires | FileCheck %s --check-prefix=TRUE
# TRUE: UNSUPPORTED{{:}} filter-requires :: base.txt
# TRUE: UNSUPPORTED{{:}} filter-requires :: half.txt
# TRUE: PASS: filter-requires :: true.txt
# TRUE: Passed{{ *}}: 1

# RUN: %{lit} -a %S/Inputs/filter-requires | FileCheck %s --check-prefix=DEFAULT
# DEFAULT: PASS: filter-requires :: base.txt
# DEFAULT: UNSUPPORTED{{:}} filter-requires :: half.txt
# DEFAULT: Test requires the following unavailable features: Half
# DEFAULT: PASS: filter-requires :: regex.txt
# DEFAULT: PASS: filter-requires :: true.txt
# DEFAULT: Passed{{ *}}: 3

# RUN: not %{lit} -a --filter-requires=Half %S/Inputs/filter-requires/regex.txt | FileCheck %s --check-prefix=REGEX
# REGEX: UNRESOLVED: filter-requires :: regex.txt
# REGEX: --filter-requires does not support '{{.*}}' patterns

# RUN: not %{lit} --filter-requires="{{.*}}" %S/Inputs/filter-requires 2>&1 | FileCheck %s --check-prefix=BAD-REGEX
# BAD-REGEX: error: argument --filter-requires: --filter-requires does not support '{{.*}}' patterns
# RUN: not %{lit} --filter-requires="Half &&" %S/Inputs/filter-requires 2>&1 | FileCheck %s --check-prefix=SYNTAX
# SYNTAX: error: argument --filter-requires: expected:
# RUN: not %{lit} --filter-requires="Half*" %S/Inputs/filter-requires 2>&1 | FileCheck %s --check-prefix=WILDCARD
# WILDCARD: error: argument --filter-requires: couldn't parse text:

# RUN: not %{lit} -a --filter-requires=Half -Dlimit=1 %S/Inputs/filter-requires/half.txt | FileCheck %s --check-prefix=LIMIT
# LIMIT: UNRESOLVED: filter-requires :: half.txt
# LIMIT: --filter-requires cannot be combined with limit_to_features
# RUN: %{lit} -a -Dlimit=1 %S/Inputs/filter-requires/half.txt | FileCheck %s --check-prefix=DEFAULT-LIMIT
# DEFAULT-LIMIT: UNSUPPORTED{{:}} filter-requires :: half.txt
# DEFAULT-LIMIT: Test requires the following unavailable features: Half

# RUN: %{lit} -a -j2 --filter-requires=Half %S/Inputs/filter-requires/half.txt %S/Inputs/filter-requires/features.txt | FileCheck %s --check-prefix=PARALLEL
# PARALLEL: Passed{{ *}}: 2

# Like REQUIRES availability checks, group matching gates execution, not discovery.
# RUN: %{lit} -a --max-tests=1 --filter-requires=Base %S/Inputs/filter-requires | FileCheck %s --check-prefix=MAX
# MAX: UNSUPPORTED{{:}} filter-requires :: and.txt
# MAX-NOT: PASS:

# RUN: %{lit} -a --filter-requires=DoesNotExist %S/Inputs/filter-requires/half.txt | FileCheck %s --check-prefix=NO-MATCH
# NO-MATCH: UNSUPPORTED{{:}} filter-requires :: half.txt
# NO-MATCH: Test REQUIRES does not match --filter-requires 'DoesNotExist'
