irdl.dialect @test_irdl_to_cpp {
    irdl.type @foo

    irdl.operation @bar {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}
