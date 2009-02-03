-- RUN: %llvmgcc -c %s
package Non_Bitfield is
   type SP is access String;
   type E is (A, B, C);
   type T (D : E) is record
      case D is
         when A => X : Boolean;
         when B => Y : SP;
         when C => Z : String (1 .. 2);
      end case;
   end record;
end;
