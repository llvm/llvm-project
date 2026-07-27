# Test the WTT (.wtl) reporter (--wtt-output).
#

# RUN: rm -f %t.wtl
# RUN: not %{lit} --wtt-output %t.wtl %{inputs}/wtt-output

# WttReport writes UTF-16; decode to UTF-8 so FileCheck can read it.
# RUN: %{python} -c "import io,sys; sys.stdout.write(io.open(r'%t.wtl', encoding='utf-16').read())" | FileCheck %s --implicit-check-not="<Err " --implicit-check-not="wtt-data :: excluded.ini"

# The .wtl must be well-formed XML (HLK/WTT Studio must be able to open it).
# RUN: %{python} -c "import xml.dom.minidom as m; m.parse(r'%t.wtl')"

# Every StartTest must have a matching EndTest (HLK rejects unbalanced logs).
# RUN: %{python} -c "import io; t=io.open(r'%t.wtl',encoding='utf-16').read(); assert t.count('<StartTest')==t.count('<EndTest'), 'unbalanced StartTest/EndTest'"

# The file must be UTF-16 (HLK expects UTF-16 encoding).
# RUN: %{python} -c "assert open(r'%t.wtl','rb').read(2)==b'\xff\xfe', 'missing UTF-16 BOM'"

# CHECK:      <?xml version="1.0" encoding="utf-16"?>
# CHECK-NEXT: <WTT-Logger>
# CHECK-NEXT: <RTI ID="" Machine=
# CHECK:      <CTX ID="{{[0-9]+}}" Current="WTTLOG" Parent="ROOT" />

# A failing test: per-test context is bound (this is what HLK uses to attribute
# results; without it every test shows up as "Unknown"). Failures use <Error>
# (not <Err>) and map to Result="Fail".
# CHECK:      <CTX ID="" Current="wtt-data :: fail.ini" Parent="WTTLOG" />
# CHECK-NEXT: <StartTest Title="wtt-data :: fail.ini" TUID="" CA="0" LA="0">
# CHECK-NEXT: <rti id="" />
# CHECK-NEXT: <ctx id="" />
# CHECK:      <Error UserText="test failed"
# CHECK:      <EndTest Title="wtt-data :: fail.ini" TUID="" Result="Fail"

# UNSUPPORTED (via a missing required feature) is reported as Pass.
# CHECK:      <EndTest Title="wtt-data :: missing_feature.ini" TUID="" Result="Pass"

# Failure output is sanitized: the newline is flattened to a space and the XML
# metacharacters are escaped so the attribute stays well-formed.
# CHECK:      <Error UserText='first line second "line" &amp; &lt;tag&gt;
# CHECK:      <EndTest Title="wtt-data :: multiline_fail.ini" TUID="" Result="Fail"

# Passing tests emit their output as a <Msg>.
# CHECK:      <Msg UserText="not shown"
# CHECK:      <EndTest Title="wtt-data :: pass.ini" TUID="" Result="Pass"
# CHECK:      <Msg UserText="ran ok"
# CHECK:      <EndTest Title="wtt-data :: pass_with_output.ini" TUID="" Result="Pass"

# UNSUPPORTED (explicit) is reported as Pass.
# CHECK:      <EndTest Title="wtt-data :: unsupported.ini" TUID="" Result="Pass"

# XFAIL is reported as Pass (WTT has no expected-failure concept).
# CHECK:      <EndTest Title="wtt-data :: xfail.ini" TUID="" Result="Pass"

# Trailing tallies and rollup. UNSUPPORTED tests fold into Passed; EXCLUDED
# tests are omitted from the log and from Total.
# CHECK:      2 test(s) were UNSUPPORTED on this device and reported as Pass
# CHECK:      1 test(s) were not run (1 excluded) and are omitted
# CHECK:      <PFRollup Total="7" Passed="5" Failed="2" Blocked="0" Warned="0" Skipped="0"
# CHECK:      </WTT-Logger>
