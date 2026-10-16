%struct.A = type { %struct.B }
%struct.B = type opaque

@g = external global %struct.A
