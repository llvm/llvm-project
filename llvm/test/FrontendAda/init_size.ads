-- RUN: %llvmgcc -c %s
package Init_Size is
   type T (B : Boolean := False) is record
      case B is
         when False =>
            I : Integer;
         when True =>
            J : Long_Long_Integer; -- Bigger than I
      end case;
   end record;
   A_T : constant T := (False, 0);
end;
