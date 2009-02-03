; RUN:  llvm-as < %s | opt -sccp | llvm-dis | not grep xor

define i11129 @test1() {
        %B = shl i11129 1, 11128 
        %C = sub i11129 %B, 1
        %D = xor i11129 %B, %C
        
	ret i11129 %D
}
