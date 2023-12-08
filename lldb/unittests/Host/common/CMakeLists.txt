set (FILES
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
  )
add_unittest_inputs(HostCommonTests "${test_inputs}")
