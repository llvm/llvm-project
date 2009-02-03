-- RUN: %llvmgcc -c %s
package Field_Order is
   type Tagged_Type is abstract tagged null record;
   type With_Discriminant (L : Positive) is new Tagged_Type with record
      S : String (1 .. L);
   end record;
end;
