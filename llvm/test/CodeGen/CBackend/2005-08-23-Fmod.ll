; RUN: llvm-as < %s | llc -march=c | grep fmod

define double @test(double %A, double %B) {
        %C = frem double %A, %B         ; <double> [#uses=1]
        ret double %C
}

