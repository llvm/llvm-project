private class V
{ 
    func layoutSubviews() {
        print("patatino") //%self.expect('expr typealias $MyV = V')
    }
}

private var my_v = V()
my_v.layoutSubviews()
