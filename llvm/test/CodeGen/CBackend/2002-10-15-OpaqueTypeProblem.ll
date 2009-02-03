; RUN: llvm-as < %s | llc -march=c

%MPI_Comm = type %struct.Comm*
%struct.Comm = type opaque
@thing = global %MPI_Comm* null         ; <%MPI_Comm**> [#uses=0]

