set (FILES
  DiagnosticsRenderingTest.cpp
  ZipFileResolverTest.cpp
)

add_lldb_unittest(HostCommonTests
  ${FILES}
  LINK_LIBS
    lldbHost
    lldbUtilityHelpers
  )

set(test_inputs
  zip-test.zip
  zip-test-no-extras.zip
  )
add_unittest_inputs(HostCommonTests "${test_inputs}")
