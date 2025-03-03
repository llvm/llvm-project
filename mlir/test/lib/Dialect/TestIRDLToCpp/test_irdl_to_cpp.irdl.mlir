irdl.dialect @test_irdl_to_cpp {
    irdl.type @foo

    irdl.operation @bar {
        %0 = irdl.any
        irdl.results(res: %0)
    }


    irdl.operation @beef {
        %0 = irdl.any
        irdl.operands(lhs: %0, rhs: %0)
        irdl.results(res: %0)
    }
}
