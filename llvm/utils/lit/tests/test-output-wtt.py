# Test the WTT (.wtl) reporter (--wtt-output).
#
# The reporter writes UTF-16, which FileCheck cannot read directly, so
# transcode the log to UTF-8 once and then assert the entire output.  The
# reporter emits tests in a stable (suite, path) order, so the log is
# deterministic.

# RUN: rm -f %t.wtl %t.wtl.utf8
# RUN: not %{lit} --wtt-output %t.wtl %{inputs}/wtt-output
# RUN: %{python} -c "import io; io.open(r'%t.wtl.utf8','w',encoding='utf-8').write(io.open(r'%t.wtl',encoding='utf-16').read())"
# RUN: FileCheck %s < %t.wtl.utf8

# The only volatile fields are the machine/pid/timestamp in the <RTI> header and
# the CA/LA tick counts (elapsed seconds, which round up on slower machines);
# everything else is fixed, so the rest of the log is asserted verbatim.

# CHECK:      <?xml version="1.0" encoding="utf-16"?>
# CHECK-NEXT: <WTT-Logger>
# CHECK-NEXT: <RTI ID="" Machine="{{.*}}" ProcessName="lit" ProcessID="{{[0-9]+}}" ThreadID="0" BaseTime="{{.*}}" Frequency="1" />
# CHECK-NEXT: <CTX ID="1" Current="WTTLOG" Parent="ROOT" />
# CHECK-NEXT: <CTX ID="" Current="wtt-data :: fail.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: fail.ini" TUID="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </StartTest>
# CHECK-NEXT: <Error UserText="test failed" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </Error>
# CHECK-NEXT: <EndTest Title="wtt-data :: fail.ini" TUID="" Result="Fail" Repro="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </EndTest>
# CHECK-NEXT: <CTX ID="" Current="wtt-data :: missing_feature.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: missing_feature.ini" TUID="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </StartTest>
# CHECK-NEXT: <Msg UserText="UNSUPPORTED on this device; reported as Pass (not applicable)." CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </Msg>
# CHECK-NEXT: <EndTest Title="wtt-data :: missing_feature.ini" TUID="" Result="Pass" Repro="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </EndTest>
# CHECK-NEXT: <CTX ID="" Current="wtt-data :: multiline_fail.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: multiline_fail.ini" TUID="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </StartTest>
# CHECK-NEXT: <Error UserText='first line second "line" &amp; &lt;tag&gt; ]]&gt;' CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </Error>
# CHECK-NEXT: <EndTest Title="wtt-data :: multiline_fail.ini" TUID="" Result="Fail" Repro="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </EndTest>
# CHECK-NEXT: <CTX ID="" Current="wtt-data :: pass.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: pass.ini" TUID="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </StartTest>
# CHECK-NEXT: <Msg UserText="not shown" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </Msg>
# CHECK-NEXT: <EndTest Title="wtt-data :: pass.ini" TUID="" Result="Pass" Repro="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </EndTest>
# CHECK-NEXT: <CTX ID="" Current="wtt-data :: pass_with_output.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: pass_with_output.ini" TUID="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </StartTest>
# CHECK-NEXT: <Msg UserText="ran ok" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </Msg>
# CHECK-NEXT: <EndTest Title="wtt-data :: pass_with_output.ini" TUID="" Result="Pass" Repro="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </EndTest>
# CHECK-NEXT: <CTX ID="" Current="wtt-data :: unsupported.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: unsupported.ini" TUID="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </StartTest>
# CHECK-NEXT: <Msg UserText="UNSUPPORTED on this device; reported as Pass (not applicable)." CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </Msg>
# CHECK-NEXT: <EndTest Title="wtt-data :: unsupported.ini" TUID="" Result="Pass" Repro="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </EndTest>
# CHECK-NEXT: <CTX ID="" Current="wtt-data :: xfail.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: xfail.ini" TUID="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </StartTest>
# CHECK-NEXT: <Msg UserText="expected fail" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </Msg>
# CHECK-NEXT: <EndTest Title="wtt-data :: xfail.ini" TUID="" Result="Pass" Repro="" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK-NEXT: </EndTest>
# CHECK-NEXT: <Msg UserText="2 test(s) were UNSUPPORTED on this device and reported as Pass (not applicable)." CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="1" />
# CHECK-NEXT: </Msg>
# CHECK-NEXT: <Msg UserText="1 test(s) were not run (1 excluded) and are omitted from the pass/fail results." CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="1" />
# CHECK-NEXT: </Msg>
# CHECK-NEXT: <PFRollup Total="7" Passed="5" Failed="2" Blocked="0" Warned="0" Skipped="0" CA="{{[0-9]+}}" LA="{{[0-9]+}}">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="1" />
# CHECK-NEXT: </PFRollup>
# CHECK-NEXT: </WTT-Logger>

# Filter nonmembers and selected-but-unsupported tests have no WTT test entries.
# RUN: %{lit} --filter-requires=Half --wtt-output %t.filtered.wtl %S/Inputs/filter-requires/half.txt %S/Inputs/filter-requires/base.txt %S/Inputs/filter-requires/unsupported.txt %S/Inputs/filter-requires/xfail.txt
# RUN: %{python} -c "import io; io.open(r'%t.filtered.utf8','w',encoding='utf-8').write(io.open(r'%t.filtered.wtl',encoding='utf-16').read())"
# RUN: FileCheck %s --check-prefix=FILTERED --implicit-check-not=base.txt --implicit-check-not=unsupported.txt --implicit-check-not="<StartTest" --implicit-check-not="<EndTest" < %t.filtered.utf8
# FILTERED: <StartTest Title="filter-requires :: half.txt"
# FILTERED: <EndTest Title="filter-requires :: half.txt" TUID="" Result="Pass"
# FILTERED: <StartTest Title="filter-requires :: xfail.txt"
# FILTERED: <EndTest Title="filter-requires :: xfail.txt" TUID="" Result="Pass"
# FILTERED: 2 test(s) were not run (1 excluded, 1 unsupported) and are omitted from the pass/fail results.
# FILTERED: <PFRollup Total="2" Passed="2" Failed="0"

# A selected test that fails during execution must still be reported.
# RUN: not %{lit} --filter-requires=Half --xfail-not=xfail.txt --wtt-output %t.failed.wtl %S/Inputs/filter-requires/xfail.txt
# RUN: %{python} -c "import io; io.open(r'%t.failed.utf8','w',encoding='utf-8').write(io.open(r'%t.failed.wtl',encoding='utf-16').read())"
# RUN: FileCheck %s --check-prefix=ATTEMPTED-FAIL < %t.failed.utf8
# ATTEMPTED-FAIL: <StartTest Title="filter-requires :: xfail.txt"
# ATTEMPTED-FAIL: <Error UserText=
# ATTEMPTED-FAIL: <EndTest Title="filter-requires :: xfail.txt" TUID="" Result="Fail"
# ATTEMPTED-FAIL: <PFRollup Total="1" Passed="0" Failed="1"

# An entirely excluded/unsupported run must not manufacture passing tests.
# RUN: %{lit} --filter-requires=Half --wtt-output %t.empty.wtl %S/Inputs/filter-requires/base.txt %S/Inputs/filter-requires/unsupported.txt
# RUN: %{python} -c "import io; io.open(r'%t.empty.utf8','w',encoding='utf-8').write(io.open(r'%t.empty.wtl',encoding='utf-16').read())"
# RUN: FileCheck %s --check-prefix=EMPTY --implicit-check-not="<StartTest" --implicit-check-not="<EndTest" --implicit-check-not=base.txt --implicit-check-not=unsupported.txt < %t.empty.utf8
# EMPTY: 2 test(s) were not run (1 excluded, 1 unsupported) and are omitted from the pass/fail results.
# EMPTY: <PFRollup Total="0" Passed="0" Failed="0"
