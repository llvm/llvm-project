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
